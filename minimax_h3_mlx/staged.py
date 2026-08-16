"""Strict component-staged text-to-video generation for MiniMax-H3 MLX."""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .adaln import ModulationCache, drop_adaln_weights
from .config import PipelineConfig
from .first_block_cache import FirstBlockCache, PRESETS as FBC_PRESETS
from .load import (
    load_audio_vae,
    load_audio_vae_config,
    load_dit,
    load_video_vae,
    load_video_vae_config,
)
from .packing import (
    AUDIO_CHANNELS,
    FPS,
    PIXEL_MEAN,
    PIXEL_STD,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from .pipeline import GenerationResult
from .scheduler import MiniMaxH3Scheduler
from .streaming_layout import is_streaming_checkpoint
from .text_encoder import MiniMaxH3TextEncoder


@dataclass
class TextArtifact:
    embeds: np.ndarray
    token_tags: np.ndarray


@dataclass
class LatentArtifact:
    video_rows: np.ndarray
    audio_rows: np.ndarray
    num_frames: int
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int
    patch_size: tuple[int, int, int]
    seconds_per_step: float


@dataclass
class StagedGenerationResult(GenerationResult):
    stage_metrics: dict[str, dict[str, object]] = field(default_factory=dict)


def _release_stage() -> float:
    gc.collect()
    mx.clear_cache()
    return mx.get_active_memory() / 1e9


def _stage_metrics(started: float, active_before_release: float) -> dict[str, object]:
    return {
        "seconds": time.perf_counter() - started,
        "peak_gb": mx.get_peak_memory() / 1e9,
        "active_before_release_gb": active_before_release,
        "active_after_release_gb": _release_stage(),
    }


def _physical_memory_gb() -> float:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9


def _resolve_qwen_stages(
    qwen_bits: int | None,
    qwen_stages: int | str,
    physical_memory_gb: float,
) -> int:
    if qwen_stages == "auto":
        if qwen_bits is not None:
            return 1
        return 2 if physical_memory_gb < 64.0 else 1
    return int(qwen_stages)


def _attention_summary(attention_backend) -> dict[str, object]:
    if attention_backend is None:
        return {"backend": "mlx_fast_sdpa"}
    return attention_backend.summary()


def _row_timestep_plan(layout, video_timesteps, audio_timesteps):
    per_step = []
    for timestep, audio_timestep in zip(video_timesteps.tolist(), audio_timesteps.tolist()):
        distinct, inverse = build_row_timesteps(
            layout,
            float(timestep),
            float(audio_timestep),
            max(float(timestep), 0.999),
            1.0,
        )
        per_step.append((np.array(distinct), np.array(inverse)))

    table = sorted({float(value) for distinct, _ in per_step for value in distinct})
    lookup = {value: index for index, value in enumerate(table)}
    plan = []
    for distinct, inverse in per_step:
        remap = np.array([lookup[float(value)] for value in distinct], dtype=np.int32)
        plan.append(mx.array(remap[inverse].astype(np.int32)))
    return mx.array(np.array(table, dtype=np.float32)), plan


def _quantize_qwen(encoder: MiniMaxH3TextEncoder, bits: int) -> int:
    count = 0

    def predicate(_path, module):
        nonlocal count
        if isinstance(module, nn.Linear) and module.weight.shape[-1] % 64 == 0:
            count += 1
            return {"group_size": 64, "bits": bits}
        return False

    nn.quantize(
        encoder.language,
        group_size=64,
        bits=bits,
        class_predicate=predicate,
    )
    mx.eval(encoder.language.parameters())
    return count


class StrictStagedTextToVideo:
    """Generate while keeping only one heavyweight component stage resident."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        transformer_dir: str | Path,
        lora_path: str | Path | None = None,
        lora_strength: float = 1.0,
        qwen_dir: str | Path | None = None,
        qwen_bits: int | None = None,
        qwen_stages: int | str = "auto",
        first_block_cache: str = "none",
        attention_backend=None,
        verbose: bool = True,
    ):
        if qwen_bits not in (None, 4, 8):
            raise ValueError(f"`qwen_bits` must be None, 4, or 8, got {qwen_bits}.")
        if qwen_stages not in ("auto", 1, 2):
            raise ValueError(f"`qwen_stages` must be auto, 1, or 2, got {qwen_stages!r}.")
        if first_block_cache not in ("none", *FBC_PRESETS):
            raise ValueError(
                f"`first_block_cache` must be none, safe, fast, or aggressive, "
                f"got {first_block_cache!r}."
            )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.transformer_dir = Path(transformer_dir)
        self.lora_path = Path(lora_path) if lora_path else None
        self.lora_strength = float(lora_strength)
        self.qwen_dir = Path(qwen_dir) if qwen_dir else self.checkpoint_dir / "text_encoder"
        self.qwen_bits = qwen_bits
        self.qwen_stages = _resolve_qwen_stages(
            qwen_bits,
            qwen_stages,
            _physical_memory_gb(),
        )
        self.first_block_cache = first_block_cache
        self.attention_backend = attention_backend
        self.verbose = bool(verbose)
        self.config = PipelineConfig.from_model_index(self.checkpoint_dir / "model_index.json")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def encode_text(self, prompt: str) -> tuple[TextArtifact, dict[str, float]]:
        if self.qwen_stages == 2:
            return self._encode_text_two_stage(prompt)

        mx.clear_cache()
        mx.reset_peak_memory()
        started = time.perf_counter()
        encoder = MiniMaxH3TextEncoder(
            self.qwen_dir,
            load_vision=False,
            verbose=self.verbose,
        )
        if encoder.quantization is not None and self.qwen_bits is not None:
            raise ValueError("`qwen_bits` must be omitted for a prequantized Qwen checkpoint.")
        prequantized = encoder.quantization is not None
        checkpoint_bits = int(encoder.quantization["bits"]) if prequantized else self.qwen_bits or 16
        quantized_linears = 0
        if self.qwen_bits is not None:
            quantized_linears = _quantize_qwen(encoder, self.qwen_bits)
            self._log(
                f"  Qwen: {quantized_linears} Linear layers quantized to "
                f"{self.qwen_bits}-bit, group 64"
            )
        embeds, token_tags = encoder.encode(prompt)
        mx.eval(embeds)
        artifact = TextArtifact(
            embeds=np.array(embeds.astype(mx.float32)),
            token_tags=np.asarray(token_tags, dtype=np.int64).copy(),
        )
        active = mx.get_active_memory() / 1e9
        del embeds, encoder
        metrics = _stage_metrics(started, active)
        metrics["quantized_linears"] = quantized_linears
        metrics["bits"] = checkpoint_bits
        metrics["prequantized"] = prequantized
        self._log(
            f"  text stage: {metrics['seconds']:.1f}s, peak {metrics['peak_gb']:.2f} GB, "
            f"after release {metrics['active_after_release_gb']:.2f} GB"
        )
        return artifact, metrics

    def _encode_text_two_stage(
        self, prompt: str
    ) -> tuple[TextArtifact, dict[str, object]]:
        from mlx_vlm.models.qwen3_vl.language import LanguageModel

        total_started = time.perf_counter()
        stage_metrics = []
        quantized_linears = 0

        mx.clear_cache()
        mx.reset_peak_memory()
        started = time.perf_counter()
        first = MiniMaxH3TextEncoder(
            self.qwen_dir,
            num_layers=25,
            layer_start=0,
            load_vision=False,
            verbose=self.verbose,
        )
        if first.quantization is not None and self.qwen_bits is not None:
            raise ValueError("`qwen_bits` must be omitted for a prequantized Qwen checkpoint.")
        prequantized = first.quantization is not None
        checkpoint_bits = int(first.quantization["bits"]) if prequantized else self.qwen_bits or 16
        if self.qwen_bits is not None:
            quantized_linears += _quantize_qwen(first, self.qwen_bits)
        input_ids, token_tags, _vision = first.build_request(prompt)
        position_ids, _ = LanguageModel.get_rope_index(
            first,
            input_ids,
            image_grid_thw=None,
            video_grid_thw=None,
            attention_mask=None,
        )
        middle = first._hidden_states(input_ids, position_ids)
        mx.eval(middle)
        middle_np = np.array(middle.astype(mx.float32))
        input_ids_np = np.array(input_ids)
        position_ids_np = np.array(position_ids)
        active = mx.get_active_memory() / 1e9
        del first, input_ids, middle, position_ids
        first_metrics = _stage_metrics(started, active)
        first_metrics["layer_range"] = [0, 25]
        stage_metrics.append(first_metrics)
        self._log(
            f"  Qwen stage 1/2: {first_metrics['seconds']:.1f}s, "
            f"peak {first_metrics['peak_gb']:.2f} GB"
        )

        mx.reset_peak_memory()
        started = time.perf_counter()
        second = MiniMaxH3TextEncoder(
            self.qwen_dir,
            num_layers=25,
            layer_start=25,
            load_vision=False,
            verbose=self.verbose,
        )
        if self.qwen_bits is not None:
            quantized_linears += _quantize_qwen(second, self.qwen_bits)
        embeds = second._hidden_states(
            mx.array(input_ids_np),
            mx.array(position_ids_np),
            inputs_embeds=mx.array(middle_np).astype(mx.bfloat16),
        )
        mx.eval(embeds)
        artifact = TextArtifact(
            embeds=np.array(embeds.astype(mx.float32)),
            token_tags=np.asarray(token_tags, dtype=np.int64).copy(),
        )
        active = mx.get_active_memory() / 1e9
        del embeds, second
        second_metrics = _stage_metrics(started, active)
        second_metrics["layer_range"] = [25, 50]
        stage_metrics.append(second_metrics)
        self._log(
            f"  Qwen stage 2/2: {second_metrics['seconds']:.1f}s, "
            f"peak {second_metrics['peak_gb']:.2f} GB"
        )

        metrics = {
            "seconds": time.perf_counter() - total_started,
            "peak_gb": max(float(item["peak_gb"]) for item in stage_metrics),
            "active_after_release_gb": float(second_metrics["active_after_release_gb"]),
            "quantized_linears": quantized_linears,
            "bits": checkpoint_bits,
            "prequantized": prequantized,
            "qwen_stages": 2,
            "stages": stage_metrics,
        }
        return artifact, metrics

    def denoise(
        self,
        text: TextArtifact,
        *,
        duration_seconds: float,
        width: int,
        height: int,
        num_inference_steps: int,
        seed: int,
        drop_adaln: bool,
    ) -> tuple[LatentArtifact, dict[str, float]]:
        if width % 32 or height % 32:
            raise ValueError(f"`height` and `width` must be multiples of 32, got {height}x{width}.")

        mx.clear_cache()
        mx.reset_peak_memory()
        started = time.perf_counter()
        streaming = is_streaming_checkpoint(self.transformer_dir)
        if streaming:
            from .streaming_dit import StreamingDiT

            dit = StreamingDiT(
                self.transformer_dir,
                lora_path=self.lora_path,
                lora_strength=self.lora_strength,
                verbose=self.verbose,
            )
        else:
            dit = load_dit(self.transformer_dir, verbose=self.verbose)
        if self.lora_path is not None and not streaming:
            from .lora import apply_lora

            apply_lora(dit, self.lora_path, self.lora_strength, verbose=self.verbose)

        video_config = load_video_vae_config(self.checkpoint_dir / "video_vae")
        num_frames = align_num_frames(int(round(duration_seconds * FPS)))
        num_latent_frames = video_latent_num_frames(num_frames)
        latent_height = height // video_config.spatial_compression_ratio
        latent_width = width // video_config.spatial_compression_ratio
        num_audio_latents = audio_latent_num_frames(num_frames)
        patch_size = dit.config.patch_size
        layout = build_packed_sequence(
            text.token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            patch_size,
        )
        if self.attention_backend is not None:
            self.attention_backend.configure_layout(layout)
            dit.set_attention_backend(self.attention_backend)
        self._log(
            f"  denoise stage: {width}x{height}, {num_frames} frames, "
            f"{layout.sequence_length:,} packed rows"
        )

        mx.random.seed(seed)
        latents = mx.random.normal(
            (1, video_config.latent_channels, num_latent_frames, latent_height, latent_width)
        ).astype(mx.float32)
        video_rows = patchify_video_latents(latents, patch_size)
        audio_config = load_audio_vae_config(self.checkpoint_dir / "audio_vae")
        audio_rows = mx.random.normal(
            (num_audio_latents * AUDIO_CHANNELS, audio_config.latent_channels)
        ).astype(mx.float32)

        video_scheduler = MiniMaxH3Scheduler(shift=self.config.sigma_shift_video)
        audio_scheduler = MiniMaxH3Scheduler(shift=self.config.sigma_shift_audio)
        video_scheduler.set_timesteps(num_inference_steps)
        audio_scheduler.set_timesteps(num_inference_steps)
        timestep_table, plan = _row_timestep_plan(
            layout, video_scheduler.timesteps, audio_scheduler.timesteps
        )
        modulation_cache = (
            dit.build_modulation_cache(timestep_table)
            if streaming
            else ModulationCache.build(dit, timestep_table, dtype=mx.bfloat16)
        )
        modulation_cache_bytes = modulation_cache.nbytes()
        adaln_freed_bytes = 0
        if drop_adaln and not streaming:
            adaln_freed_bytes = drop_adaln_weights(dit)
            mx.eval(dit.parameters())

        embeds = mx.array(text.embeds).astype(mx.bfloat16)
        block_cache = (
            None
            if self.first_block_cache == "none"
            else FirstBlockCache(FBC_PRESETS[self.first_block_cache])
        )
        step_times = []
        for index, timestep in enumerate(video_scheduler.timesteps.tolist()):
            step_started = time.perf_counter()
            if block_cache is not None:
                block_cache.begin_step(index, len(video_scheduler.timesteps))
            if self.attention_backend is not None:
                self.attention_backend.begin_step(index, len(video_scheduler.timesteps))
            video_prediction, audio_prediction = dit(
                video_rows[None].astype(mx.bfloat16),
                audio_rows[None].astype(mx.bfloat16),
                embeds,
                timestep_table,
                plan[index],
                layout.token_tags,
                layout.position_ids,
                layout.video_indices,
                layout.audio_indices,
                layout.text_indices,
                modulation_cache=modulation_cache,
                first_block_cache=block_cache,
            )
            video_rows = video_scheduler.step(
                video_prediction[0].astype(mx.float32), float(timestep), video_rows
            )
            audio_rows = audio_scheduler.step(
                audio_prediction[0].astype(mx.float32),
                float(audio_scheduler.timesteps[index].item()),
                audio_rows,
            )
            mx.eval(video_rows, audio_rows)
            if not mx.all(mx.isfinite(video_rows)).item():
                raise FloatingPointError(f"Non-finite video latent after denoise step {index + 1}.")
            if not mx.all(mx.isfinite(audio_rows)).item():
                raise FloatingPointError(f"Non-finite audio latent after denoise step {index + 1}.")
            step_times.append(time.perf_counter() - step_started)
            self._log(
                f"    step {index + 1}/{len(video_scheduler.timesteps)}: "
                f"{step_times[-1]:.2f}s"
            )

        artifact = LatentArtifact(
            video_rows=np.array(video_rows.astype(mx.float32)),
            audio_rows=np.array(audio_rows.astype(mx.float32)),
            num_frames=num_frames,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            patch_size=patch_size,
            seconds_per_step=sum(step_times) / max(len(step_times), 1),
        )
        active = mx.get_active_memory() / 1e9
        streaming_summary = dit.summary() if streaming else None
        del (
            audio_prediction,
            audio_rows,
            dit,
            embeds,
            latents,
            layout,
            modulation_cache,
            plan,
            timestep_table,
            video_prediction,
            video_rows,
        )
        metrics = _stage_metrics(started, active)
        metrics["forward_count"] = len(step_times)
        metrics["modulation_cache_mb"] = modulation_cache_bytes / 1e6
        metrics["adaln_freed_gb"] = adaln_freed_bytes / 1e9
        if streaming_summary is not None:
            metrics["streaming"] = streaming_summary
        if block_cache is not None:
            metrics["first_block_cache"] = block_cache.summary()
        metrics["attention_backend"] = _attention_summary(self.attention_backend)
        self._log(
            f"  denoise release: peak {metrics['peak_gb']:.2f} GB, "
            f"after release {metrics['active_after_release_gb']:.2f} GB"
        )
        return artifact, metrics

    def decode_video(
        self, latents: LatentArtifact
    ) -> tuple[np.ndarray, dict[str, float]]:
        mx.clear_cache()
        mx.reset_peak_memory()
        started = time.perf_counter()
        vae = load_video_vae(self.checkpoint_dir / "video_vae")
        config = vae.config
        rows = mx.array(latents.video_rows)
        values = unpatchify_video_tokens(
            rows,
            latents.num_latent_frames,
            latents.latent_height,
            latents.latent_width,
            config.latent_channels,
            latents.patch_size,
        )
        mean = mx.array(np.array(config.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
        std = mx.array(np.array(config.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
        decoded = np.array(vae.decode((values * std + mean).astype(mx.float32)))
        pixel_mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
        pixel_std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
        decoded = np.clip(decoded * pixel_std + pixel_mean, 0.0, 1.0)
        video = (decoded[0].transpose(1, 2, 3, 0) * 255.0 + 0.5).astype(np.uint8)
        active = mx.get_active_memory() / 1e9
        del decoded, rows, vae, values
        metrics = _stage_metrics(started, active)
        self._log(
            f"  video VAE: {metrics['seconds']:.1f}s, peak {metrics['peak_gb']:.2f} GB, "
            f"after release {metrics['active_after_release_gb']:.2f} GB"
        )
        return video, metrics

    def decode_audio(
        self, latents: LatentArtifact
    ) -> tuple[np.ndarray, int, dict[str, float]]:
        mx.clear_cache()
        mx.reset_peak_memory()
        started = time.perf_counter()
        vae = load_audio_vae(self.checkpoint_dir / "audio_vae")
        config = vae.config
        rows = mx.array(latents.audio_rows)
        values = unpack_audio_tokens(rows, latents.num_audio_latents)
        mean = mx.array(np.array(config.latents_mean, np.float32)).reshape(1, -1, 1)
        std = mx.array(np.array(config.latents_std, np.float32)).reshape(1, -1, 1)
        waveform = np.array(vae.decode((values * std + mean).astype(mx.float32)))[:, 0, :]
        audio = waveform.astype(np.float32)
        sample_rate = config.sampling_rate
        active = mx.get_active_memory() / 1e9
        del rows, vae, values, waveform
        metrics = _stage_metrics(started, active)
        self._log(
            f"  audio VAE: {metrics['seconds']:.1f}s, peak {metrics['peak_gb']:.2f} GB, "
            f"after release {metrics['active_after_release_gb']:.2f} GB"
        )
        return audio, sample_rate, metrics

    def __call__(
        self,
        prompt: str,
        *,
        duration_seconds: float = 5.0,
        width: int = 864,
        height: int = 480,
        num_inference_steps: int = 5,
        seed: int = 0,
        drop_adaln: bool = True,
    ) -> StagedGenerationResult:
        started = time.perf_counter()
        text, text_metrics = self.encode_text(prompt)
        latents, denoise_metrics = self.denoise(
            text,
            duration_seconds=duration_seconds,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            seed=seed,
            drop_adaln=drop_adaln,
        )
        del text
        video, video_metrics = self.decode_video(latents)
        audio, sample_rate, audio_metrics = self.decode_audio(latents)
        return StagedGenerationResult(
            video=video,
            audio=audio,
            sample_rate=sample_rate,
            fps=FPS,
            seconds_per_step=latents.seconds_per_step,
            total_seconds=time.perf_counter() - started,
            stage_metrics={
                "text": text_metrics,
                "denoise": denoise_metrics,
                "video_vae": video_metrics,
                "audio_vae": audio_metrics,
            },
        )


__all__ = [
    "LatentArtifact",
    "StagedGenerationResult",
    "StrictStagedTextToVideo",
    "TextArtifact",
]