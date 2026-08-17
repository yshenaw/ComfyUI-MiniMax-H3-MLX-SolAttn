# ComfyUI MiniMax H3 MLX Sol-Attn

Apple Silicon ComfyUI node for MiniMax H3 synchronized video and audio generation. It runs Qwen, the MLX DiT, Video VAE, and Audio VAE as strict stages and includes the tiled Metal Sol-Attn backend.

> Powered by MiniMax H3.

## Which model should I install?

The node supports quantized fast profiles and BF16 quality profiles. Setup downloads **4-bit only by default**.

| Profile | Transformer download | Resident after AdaLN drop | Use case |
| --- | ---: | ---: | --- |
| 4-bit core + 8-bit full AdaLN | about 25.3 GB | 11.45 GB | Recommended default; lower memory and faster linear layers |
| 8-bit core + 8-bit full AdaLN | about 35.3 GB | 21.47 GB | Higher fidelity; 64 GB+ recommended |
| 4-bit core + pruned 8-bit AdaLN | about 11.5 GB | 11.33 GB | 32 GB resident default |
| 8-bit core + pruned 8-bit AdaLN | about 21.5 GB | 21.35 GB | 48 GB resident default |
| BF16 core + rank-8 pruned AdaLN | about 40.2 GB | 40.2 GB | 48/64 GB high-quality stream2 tier |
| BF16 core + full-width AdaLN | about 66.3 GB | 40.3 GB | 96 GB+ full-quality tier |

The published 4/8-bit profiles keep AdaLN full-width at 8-bit. The BF16-pruned profile uses the official rank-8 curve checkpoint. Comfy checkpoints store QKV rows as `[Q][K][V]`; the MLX repacker converts them to the per-head interleaved layout expected by this runtime. Before that conversion was added, incorrect QKV rows produced gray output that was mistakenly attributed to pruned AdaLN. Corrected BF16-pruned resident and stream2 runs both generate normal video and audio.

The setup downloads a pinned six-shard subset of `lmstudio-community/Qwen3-VL-32B-Instruct-MLX-8bit`: embedding plus layers 0-49, exactly the part H3 evaluates. Cross-layer unquantized norm tensors are byte-identical to the MiniMax H3 Qwen checkpoint. The unused final 14 layers, LM head, and seventh shard are not downloaded.

Budget roughly **70 GB free for the default 4-bit setup**, or **105 GB for both transformer profiles**. This includes the 32.13 GB prequantized Qwen8 subset and avoids the roughly 65 GB upstream BF16 text encoder.

## Requirements

- Apple Silicon Mac running macOS
- ComfyUI with Python 3.11+
- About 70 GB free disk space for the default profile
- 48 GB unified memory for the default 4-bit profile; 64 GB+ for a comfortable 8-bit profile
- Internet access for the initial model download

The tiled Sol bridge uses ComfyUI's MPS-enabled PyTorch. MLX activations and Q/K/V are BF16; online softmax and scheduler updates accumulate in FP32.

## Install

Install this repository through ComfyUI Manager, then restart ComfyUI. For a manual install:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yshenaw/ComfyUI-MiniMax-H3-MLX-SolAttn.git
cd ComfyUI-MiniMax-H3-MLX-SolAttn
python3 -m pip install -r requirements.txt
```

Download the recommended 4-bit model set with one command:

```bash
python3 scripts/setup_comfyui.py
```

The command asks you to accept the MiniMax H3 Community License, then installs models under `ComfyUI/models/minimax_h3`. Downloads are resumable through Hugging Face Hub.

Optional profiles:

```bash
# Install the 8-bit transformer instead of 4-bit
python3 scripts/setup_comfyui.py --profile 8

# Install the BF16-pruned quality profile for stream2
python3 scripts/setup_comfyui.py --profile bf16-pruned

# Build resident pruned quant profiles locally from the shared BF16 source
python3 scripts/setup_comfyui.py --profile 4-pruned
python3 scripts/setup_comfyui.py --profile 8-pruned

# Install both transformer profiles
python3 scripts/setup_comfyui.py --profile all

# Non-interactive license confirmation for automation
python3 scripts/setup_comfyui.py --profile 4 --accept-license
```

For unusual ComfyUI layouts, pass `--models-dir /path/to/ComfyUI/models` or set `MINIMAX_H3_MODELS_DIR` before starting ComfyUI.

The pruned quant profiles are never downloaded as third-party converted weights. Setup downloads the official BF16-pruned checkpoint and converts one block at a time. Measured converter peak RSS was 3.43 GB; Full4-pruned took 65 seconds and Full8-pruned 94 seconds on the M3 Ultra test system.

## Use

1. Restart ComfyUI after setup.
2. Load `workflows/minimax_h3_mlx_turbo4_sol.json` from this repository.
3. Edit the prompt and queue the workflow.

The workflow saves an MP4 and WAV using ComfyUI's core `CreateVideo`, `SaveVideo`, and `SaveAudio` nodes. Its defaults are the tested 48 GB configuration:

- DiT: 4-bit core with full-width 8-bit AdaLN
- Qwen: prequantized 8-bit, independently staged as layers 0-24 and 25-49
- Sampler: Turbo4, four model forwards
- Attention: tiled Metal Sol-Attn, sparse middle 20%-90%
- Canvas: 864x480, 5 seconds, 24 fps
- First Block Cache: disabled for Turbo4

The generator returns standard ComfyUI `IMAGE`, `AUDIO`, and JSON stats outputs. It unloads other ComfyUI Torch models before generation so Torch and MLX do not retain unrelated weights in unified memory.

The same node exposes three workflow presets:

| Preset | Forwards | Turbo LoRA | FBC | Sol-Attn default |
| --- | ---: | --- | --- | --- |
| Turbo 4 Fast | 4 | on | off | on |
| Turbo 6 Balanced | 6 | on | off | on |
| Quality 20 | 20 | off | safe | on |

`memory_mode=auto` uses five-block streaming below 40 GB and resident DiT weights otherwise. Prequantized Qwen8 uses 25+25 stages in the streaming tier and a single 50-layer stage in resident tiers.

## Measured result

All timing numbers below were measured on a Mac Studio with an M3 Ultra, 80-core GPU, 512 GB unified memory, and 819 GB/s memory bandwidth. They are not Mac mini timings. The memory figures are still useful for capacity planning, but absolute speed does not transfer directly to an M4 Pro Mac mini.

At 864x480, Turbo4 with the 4-bit profile produced normal video and audio:

| Metric | Sol-Attn | Dense |
| --- | ---: | ---: |
| Denoise | 155.01 s | 173.09 s |
| Total wall time | 225.86 s | 244.25 s |
| MLX denoise peak | 26.14 GB | 26.14 GB |
| Process footprint | 35.43 GB | 34.36 GB |
| Swap growth | 0 GB | 0 GB |

Sol-Attn reduced denoise time by 10.45% on the M3 Ultra. With roughly 8 GB reserved for macOS and ComfyUI, the 4-bit resident profile leaves useful memory headroom on a 48 GB Mac. The full8 profile reached a 44.02 GB process footprint in a clean run, so it is experimental on 48 GB and recommended only at 64 GB or above.

A 48 GB Mac mini uses M4 Pro, with up to a 20-core GPU and 273 GB/s memory bandwidth. Based on the 4x GPU-core and 3x bandwidth gap, moderated by newer M4 GPU cores, expect roughly 2.5-3.5x the M3 Ultra wall time until a real M4 Pro benchmark is available. The five-second Full4 Turbo4 workflow is therefore estimated at about 9-14 minutes on a 20-core M4 Pro Mac mini; the 16-core GPU configuration may take longer. This is an estimate, not a measured result.

## Experimental block streaming

The exact Full4 checkpoint can be repacked so fixed modules stay resident while the 50 main DiT blocks load in five-block chunks. AdaLN chunks are read once to build the full-width BF16 modulation cache; only core chunks are reread on every denoise forward.

```bash
python3 scripts/build_streaming_checkpoint.py \
  --source /path/to/MiniMax-H3-MLX-4bit \
  --out /path/to/MiniMax-H3-MLX-4bit-stream5 \
  --chunk-size 5
```

Pass the resulting directory as `--transformer`. The staged runner detects `streaming_manifest.json` automatically. Streaming supports the same Safe First Block Cache used by Quality20.

Measured on the same 80-core M3 Ultra at 864x480, five seconds, Qwen8, Turbo4, and tiled Metal Sol-Attn:

| Metric | Full4 resident | Full4 stream5 |
| --- | ---: | ---: |
| Denoise | 155.01 s | 159.92 s |
| Wall time | 225.86 s | 230.96 s |
| Denoise MLX peak | 26.14 GB | 6.18 GB |
| Active before denoise release | 12.16 GB | 1.48 GB |
| Process peak RSS | 26.60 GB | 26.42 GB |
| Process peak footprint | 35.43 GB | 39.73 GB |

Streaming reduced the DiT MLX peak by 76.37% and increased denoise time by 3.17% in a warm filesystem-cache run. All four forwards read 43.36 GB of core tensors in total. The streamed and resident MP4 and WAV outputs were byte-for-byte identical.

Replacing runtime Qwen quantization with the pinned prequantized Qwen8 subset reduced the Qwen peak from 27.11 GB to 13.83 GB. The complete prequantized-Qwen8 + Full4-stream5 + Turbo4 + Sol run measured:

| Metric | Prequantized Qwen8 + Full4 stream5 |
| --- | ---: |
| Text peak | 13.83 GB |
| Denoise peak | 6.18 GB |
| Process RSS | 13.61 GB |
| Process footprint | 25.77 GB |
| Wall time | 233.24 s |
| Swap growth | 0 GB |

The generated video was normal and dynamic; audio waveform correlation against the runtime-quantized baseline was 0.9844. This makes 32 GB a viable experimental memory target, not a speed claim for 32 GB hardware. The standard resident Full4 profile remains the recommended 48 GB configuration.

The repaired BF16-pruned stream2 profile was measured with prequantized Qwen8, Turbo4, and Sol-Attn:

| Metric | BF16-pruned stream2 |
| --- | ---: |
| Wall time | 234.30 s |
| Denoise | 171.88 s |
| Denoise MLX peak | 7.63 GB |
| Process footprint | 24.70 GB |
| Core data read | 154.15 GB |
| Swap growth | 0 GB |

The corrected resident and stream2 videos both contain the expected moving sports car and neon city. Audio waveform correlation between them was 0.906. Timings are from the 80-core M3 Ultra test machine; 48/64 GB systems should treat BF16 stream2 as a quality-first, SSD-intensive mode.

## Command-line generation

The same staged backend can run outside ComfyUI:

```bash
python3 scripts/generate_staged.py \
  "Cinematic tracking shot of a red sports car in a neon city" \
  --checkpoint /path/to/models/minimax_h3/upstream/FL2VA \
  --transformer /path/to/models/minimax_h3/transformers/4-bit \
  --qwen-dir /path/to/models/minimax_h3/qwen/8-bit \
  --qwen-stages 2 \
  --lora /path/to/models/minimax_h3/loras/minimax_h3_turbo_4step_ema_ckpt850.safetensors \
  --steps 5 --attention-backend torch-mps \
  --sol-attn-mps-dir ./sol_attn_mps \
  --duration 5 --width 864 --height 480 \
  --output out.mp4
```

## Development

```bash
python3 -m pytest -q \
  tests/test_comfy_models.py \
  tests/test_comfyui_nodes.py \
  tests/test_setup_comfyui.py \
  tests/test_comfy_workflow.py \
  tests/test_mlx_sol_attn.py \
  tests/test_staged.py
```

Spectrum Sampling is not implemented. The production sparse backend is MLX to DLPack to PyTorch MPS tiled Metal; the pure MLX implementation is a numerical reference only.

## License

Repository source code is Apache-2.0. Model weights are not included in Git and are governed separately:

- MiniMax H3 and converted MLX transformer weights: MiniMax H3 Community License
- Turbo LoRA: Apache-2.0 according to its Hugging Face repository
- Sol-Attn adaptations: see `THIRD_PARTY_NOTICES.md`

Review the MiniMax license before downloading. Generated experiences must retain the required "Powered by MiniMax H3" attribution.
