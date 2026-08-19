"""Exact MiniMax-H3 DiT execution with block weights streamed from disk."""

from __future__ import annotations

import gc
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .adaln import ModulationCache
from .config import MODALITY_NUM, DiTConfig
from .dit import AdaLayerNormModulation, MiniMaxH3DiT, TransformerBlock, param_dtype
from .lora import CurveLoRALinear, LoRALinear, _linear_input_dim
from .offset_io import RawParts, raw_parts_to_mlx, read_safetensors_parts
from .quantize import QuantConfig, apply_quantization_structure
from .streaming_layout import load_manifest


_CORE_LINEAR_PATHS = {
    "attn.qkv_proj",
    "attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
}


def _load_parts(model_dir: Path, names: list[str]) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for name in names:
        weights.update(mx.load(str(model_dir / name)))
    return weights


def _update_strict(module: nn.Module, weights: dict[str, mx.array], prefix: str) -> None:
    local = {
        key[len(prefix) :]: value
        for key, value in weights.items()
        if key.startswith(prefix)
    }
    expected = {key for key, _ in tree_flatten(module.parameters())}
    missing = sorted(expected - local.keys())
    unexpected = sorted(local.keys() - expected)
    if missing or unexpected:
        raise KeyError(
            f"Streaming checkpoint mismatch for {prefix or 'fixed'}: "
            f"{len(missing)} missing (e.g. {missing[:4]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:4]})."
        )
    module.update(tree_unflatten(list(local.items())))


def _apply_lora_subset(
    module: nn.Module,
    weights: dict[str, mx.array],
    *,
    source_prefix: str,
    include,
    strength: float,
    allow_curve: bool = False,
) -> int:
    modules = dict(module.named_modules())
    targets = sorted(
        {
            key.rsplit(".lora_", 1)[0]
            for key in weights
            if key.startswith(source_prefix) and include(key.rsplit(".lora_", 1)[0])
        }
    )
    replacements = []
    for source_name in targets:
        local_name = source_name[len(source_prefix) :]
        base = modules.get(local_name)
        if base is None:
            raise KeyError(f"Streaming LoRA target not found: {source_name} -> {local_name}")
        lora_a = weights[f"{source_name}.lora_A.weight"]
        lora_b = weights[f"{source_name}.lora_B.weight"]
        if _linear_input_dim(base) == lora_a.shape[-1]:
            replacement = LoRALinear(base, lora_a, lora_b, strength)
        elif allow_curve:
            replacement = CurveLoRALinear(base, lora_a, lora_b, strength)
        else:
            raise ValueError(
                f"Streaming LoRA input mismatch for {source_name}: "
                f"base={_linear_input_dim(base)}, adapter={lora_a.shape[-1]}."
            )
        replacements.append((local_name, replacement))
    if replacements:
        module.update_modules(tree_unflatten(replacements))
    return len(replacements)


class StreamingDiT:
    """Keep fixed DiT modules resident and stream main blocks chunk by chunk."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        lora_path: str | Path | None = None,
        lora_strength: float = 1.0,
        core_io: str | None = None,
        verbose: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.manifest = load_manifest(self.model_dir)
        self.config = DiTConfig.from_json(self.model_dir / "config.json")
        curve_shape = self.manifest.get("adaln_curve_shape")
        self.curve_grid = int(curve_shape[0]) if curve_shape else None
        self.curve_dim = int(curve_shape[1]) if curve_shape else None
        quant_path = self.model_dir / "quant_config.json"
        self.quant_config = None
        if quant_path.is_file():
            with open(quant_path) as handle:
                recipe = json.load(handle)
            self.quant_config = QuantConfig(
                bits=recipe["bits"],
                group_size=recipe["group_size"],
                quantize_adaln=recipe.get("quantize_adaln", False),
                adaln_bits=recipe.get("adaln_bits") or 8,
                overrides={
                    str(path): None if bits is None else int(bits)
                    for path, bits in recipe.get("overrides", {}).items()
                },
                group_overrides={
                    str(path): int(group_size)
                    for path, group_size in recipe.get("group_overrides", {}).items()
                },
            )
        self.verbose = bool(verbose)
        self.core_io = core_io or os.environ.get("MINIMAX_H3_STREAM_IO", "mlx")
        if self.core_io not in ("mlx", "offset"):
            raise ValueError(f"Unsupported streaming core I/O mode: {self.core_io!r}.")
        self.attention_backend = None
        self.core_load_seconds = 0.0
        self.core_pread_seconds = 0.0
        self.core_wait_seconds = 0.0
        self.core_convert_seconds = 0.0
        self.core_bytes_read = 0
        self.core_file_bytes_read = 0
        self.core_nocache_files = 0
        self.adaln_load_seconds = 0.0
        self.adaln_bytes_read = 0
        self.chunk_loads = 0

        fixed = MiniMaxH3DiT(
            self.config,
            adaln_curve_grid=self.curve_grid,
            adaln_curve_dim=self.curve_dim or 8,
        )
        fixed.blocks = []
        gc.collect()
        if self.quant_config is not None:
            apply_quantization_structure(fixed, self.quant_config)
        fixed_weights = _load_parts(self.model_dir, self.manifest["fixed_files"])
        _update_strict(fixed, fixed_weights, "")
        del fixed_weights
        if self.curve_grid is not None:
            curve_file = self.model_dir / self.manifest["full_curve_file"]
            fixed._adaln_lora_t_table = mx.load(str(curve_file))["silu_t_emb_grid"]

        self.lora_weights = mx.load(str(Path(lora_path))) if lora_path is not None else {}
        curve_lora = _apply_lora_subset(
            fixed,
            self.lora_weights,
            source_prefix="",
            include=lambda name: not name.startswith("blocks."),
            strength=lora_strength,
            allow_curve=self.curve_grid is not None,
        )
        if self.curve_grid is not None and curve_lora:
            fixed._adaln_curve_lora_enabled = True
        mx.eval(fixed.parameters())
        self.fixed = fixed
        self.lora_strength = float(lora_strength)

    def set_attention_backend(self, backend) -> None:
        self.attention_backend = backend

    def _new_adaln(self, block_index: int, weights: dict[str, mx.array]):
        projection = AdaLayerNormModulation(
            self.config,
            input_dim=self.curve_dim,
            apply_silu=self.curve_grid is None,
        )
        if self.quant_config is not None and self.quant_config.quantize_adaln:
            nn.quantize(
                projection,
                group_size=self.quant_config.group_size,
                bits=self.quant_config.adaln_bits,
                class_predicate=lambda path, module: (
                    {
                        "group_size": self.quant_config.group_size,
                        "bits": self.quant_config.adaln_bits,
                    }
                    if path == "linear" and isinstance(module, nn.Linear)
                    else False
                ),
            )
        prefix = f"blocks.{block_index}.adaln_proj."
        _update_strict(projection, weights, prefix)
        _apply_lora_subset(
            projection,
            self.lora_weights,
            source_prefix=prefix,
            include=lambda _name: True,
            strength=self.lora_strength,
            allow_curve=self.curve_grid is not None,
        )
        return projection

    def build_modulation_cache(self, timesteps: mx.array) -> ModulationCache:
        temb = self.fixed.modulation_input(timesteps)
        mx.eval(temb)
        tables: list[tuple[mx.array, ...] | None] = [None] * self.config.num_layers
        for chunk in self.manifest["chunks"]:
            started = time.perf_counter()
            weights = _load_parts(self.model_dir, chunk["adaln_files"])
            self.adaln_load_seconds += time.perf_counter() - started
            self.adaln_bytes_read += int(chunk["adaln_bytes"])
            for block_index in range(int(chunk["start"]), int(chunk["end"])):
                projection = self._new_adaln(block_index, weights)
                table = tuple(value.astype(mx.bfloat16) for value in projection(temb))
                mx.eval(table)
                tables[block_index] = table
                del projection
            del weights
            gc.collect()
            mx.clear_cache()
        if any(table is None for table in tables):
            raise RuntimeError("Streaming AdaLN cache is incomplete.")
        return ModulationCache(tables, timesteps)

    def _new_block(self, block_index: int, weights: dict[str, mx.array]):
        block = TransformerBlock(self.config)
        block.adaln_proj.linear = None
        if self.quant_config is not None:
            nn.quantize(
                block,
                group_size=self.quant_config.group_size,
                bits=self.quant_config.bits,
                class_predicate=lambda path, module: (
                    {
                        "group_size": self.quant_config.group_size,
                        "bits": self.quant_config.bits,
                    }
                    if path in _CORE_LINEAR_PATHS and isinstance(module, nn.Linear)
                    else False
                ),
            )
        prefix = f"blocks.{block_index}."
        _update_strict(block, weights, prefix)
        _apply_lora_subset(
            block,
            self.lora_weights,
            source_prefix=prefix,
            include=lambda name: ".adaln_proj." not in name,
            strength=self.lora_strength,
        )
        block.attn._attention_backend = self.attention_backend
        mx.eval(block.parameters())
        return block

    def __call__(
        self,
        video_latents: mx.array,
        audio_latents: mx.array,
        text_embeds: mx.array,
        timestep: mx.array,
        timestep_indices: mx.array,
        token_tags: mx.array,
        position_ids: mx.array,
        video_indices: mx.array,
        audio_indices: mx.array,
        text_indices: mx.array,
        *,
        modulation_cache: ModulationCache,
        first_block_cache=None,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        rotary = self.fixed.rope(position_ids)
        video = self.fixed.video_patch_proj(video_latents.astype(param_dtype(self.fixed.video_patch_proj)))
        audio = self.fixed.audio_patch_proj(audio_latents.astype(param_dtype(self.fixed.audio_patch_proj)))
        text = self.fixed.condition_proj(text_embeds.astype(param_dtype(self.fixed.condition_proj)))
        text = self.fixed.token_refiner(text)
        x = mx.zeros((text.shape[0], position_ids.shape[0], text.shape[-1]), dtype=text.dtype)
        x[:, text_indices] = text
        x[:, video_indices] = video.astype(text.dtype)
        x[:, audio_indices] = audio.astype(text.dtype)

        temb = self.fixed.modulation_input(timestep)
        adaln_indices = timestep_indices * MODALITY_NUM + mx.maximum(token_tags, 0)
        first_input = x if first_block_cache is not None else None
        cache_hit = False
        chunks = self.manifest["chunks"]
        executor = None
        future: Future[RawParts] | None = None
        if self.core_io == "offset" and chunks:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-offset-prefetch")
            future = executor.submit(
                read_safetensors_parts,
                self.model_dir,
                chunks[0]["core_files"],
                nocache=True,
            )
        try:
            for chunk_index, chunk in enumerate(chunks):
                delay_prefetch = chunk_index == 0 and first_block_cache is not None
                if future is None:
                    started = time.perf_counter()
                    weights = _load_parts(self.model_dir, chunk["core_files"])
                    self.core_load_seconds += time.perf_counter() - started
                else:
                    started = time.perf_counter()
                    raw = future.result()
                    self.core_wait_seconds += time.perf_counter() - started
                    self.core_pread_seconds += raw.read_seconds
                    self.core_file_bytes_read += raw.bytes_read
                    self.core_nocache_files += raw.nocache_files
                    if not delay_prefetch and chunk_index + 1 < len(chunks):
                        future = executor.submit(
                            read_safetensors_parts,
                            self.model_dir,
                            chunks[chunk_index + 1]["core_files"],
                            nocache=True,
                        )
                    else:
                        future = None
                    started = time.perf_counter()
                    weights = raw_parts_to_mlx(raw)
                    self.core_convert_seconds += time.perf_counter() - started
                    self.core_load_seconds = self.core_wait_seconds + self.core_convert_seconds
                    del raw
                self.core_bytes_read += int(chunk["core_bytes"])
                self.chunk_loads += 1
                for block_index in range(int(chunk["start"]), int(chunk["end"])):
                    block = self._new_block(block_index, weights)
                    x = block(x, modulation_cache.get(block_index), adaln_indices, rotary, mask)
                    mx.eval(x)
                    del block
                    if block_index == 0 and first_block_cache is not None:
                        cached = first_block_cache.after_first_block(first_input, x)
                        if cached is not None:
                            x = cached
                            cache_hit = True
                            break
                del weights
                gc.collect()
                mx.clear_cache()
                if cache_hit:
                    break
                if delay_prefetch and chunk_index + 1 < len(chunks):
                    future = executor.submit(
                        read_safetensors_parts,
                        self.model_dir,
                        chunks[chunk_index + 1]["core_files"],
                        nocache=True,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

        if first_block_cache is not None and not cache_hit:
            first_block_cache.finish_full_step(x)

        x = self.fixed.final_layer.norm_out(x, temb, timestep_indices)
        video_out = self.fixed.final_layer.video_out(x.astype(param_dtype(self.fixed.final_layer.video_out)))
        audio_out = self.fixed.final_layer.audio_out(x.astype(param_dtype(self.fixed.final_layer.audio_out)))
        return video_out[:, video_indices], audio_out[:, audio_indices]

    def summary(self) -> dict[str, object]:
        return {
            "format": self.manifest["format"],
            "quantized": self.quant_config is not None,
            "adaln_curve_dim": self.curve_dim,
            "chunk_size": self.manifest["chunk_size"],
            "core_io": self.core_io,
            "chunk_loads": self.chunk_loads,
            "core_gb_read": self.core_bytes_read / 1e9,
            "core_load_seconds": self.core_load_seconds,
            "core_pread_seconds": self.core_pread_seconds,
            "core_wait_seconds": self.core_wait_seconds,
            "core_convert_seconds": self.core_convert_seconds,
            "core_file_gb_read": self.core_file_bytes_read / 1e9,
            "core_nocache_files": self.core_nocache_files,
            "adaln_gb_read": self.adaln_bytes_read / 1e9,
            "adaln_load_seconds": self.adaln_load_seconds,
        }


__all__ = ["StreamingDiT"]