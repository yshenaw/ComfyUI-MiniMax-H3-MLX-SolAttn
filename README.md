# MiniMax H3 Creator for Mac — MLX + Sol-Attn

Create short videos with native synchronized audio in ComfyUI on Apple Silicon. Pick a memory tier,
write one prompt, and render directly to MP4. The node stages Qwen, DiT, Video VAE, and Audio VAE so
large components do not stay loaded together.

> Powered by MiniMax H3.

## Start here

### 1. Choose your Mac

| Unified memory | Creator preset | Setup command | Start with |
| ---: | --- | --- | --- |
| **24 GB** | Experimental Preview | `python3 scripts/setup_comfyui.py --profile 24` | 480p · 5 s · Turbo4 |
| **32 GB** | Low Memory | `python3 scripts/setup_comfyui.py --profile 4-pruned` | 480p · 5 s · Turbo4 |
| **48 GB** | Balanced | `python3 scripts/setup_comfyui.py --profile 8-pruned` | 480p · 5–15 s |
| **64 GB+** | High Quality | `python3 scripts/setup_comfyui.py --profile attention16-mlp8` | 480p first, then 720p |

For 96 GB and larger PyTorch BF16 workflows, use
[`MacOS-H3-Speedrun`](https://github.com/EvolvingLMMs-Lab/MacOS-H3-Speedrun).

### 2. Choose how long you want to wait

| Mode | Best for | Model forwards | Notes |
| --- | --- | ---: | --- |
| **Turbo 4 Fast** | Prompt drafts and most creator work | 4 | Recommended starting point |
| **Turbo 8 Balanced** | Optional second pass | 8 | Slower; perceptual validation is still limited |
| **Full 20 Quality** | Non-Turbo final comparison | 20 | Safe FBC and Sol-Attn default on |

Sol-Attn is optional and enabled by default. Full20 exposes an FBC toggle; Turbo modes ignore FBC.

### Can I use my other ComfyUI LoRAs?

Not directly through the current MLX node. The packaged workflow applies the tested MiniMax-H3
Turbo LoRA internally; it does not output a standard ComfyUI `MODEL` object, so generic `Load LoRA`
or LoRA-stack nodes cannot be inserted in front of it.

| LoRA type | Current MLX workflow | What to expect |
| --- | --- | --- |
| Packaged MiniMax-H3 Turbo LoRA | **Yes** | Applied automatically in Turbo4 and Turbo8 |
| Community LoRA trained specifically for MiniMax-H3 | Conversion required | It must match H3 module names, tensor dimensions, and base checkpoint |
| Standard ComfyUI `Load LoRA` / LoRA-stack node | **No direct connection** | The MLX generator is a complete staged node, not a ComfyUI `MODEL` pipeline |
| SD, SDXL, Flux, Wan, or another model family's LoRA | **No** | LoRAs are architecture-specific |
| Multiple LoRAs at once | Not yet | The MLX runtime currently accepts one adapter path |

The MLX backend itself can apply a matching adapter to BF16, 4-bit, 8-bit, or mixed-precision H3
linear layers. Community files often use Diffusers or ComfyUI key prefixes and therefore need a
converter before they can load. A LoRA that modifies full-width AdaLN also needs the aligned H3
timestep table when used with a pruned-AdaLN base.

Do not assume that a normal style or character LoRA can use Turbo4. Turbo4 and Turbo8 require a
LoRA trained or distilled for that short schedule. A converted non-Turbo community LoRA should be
validated with Full20 first. Community-LoRA selection, key conversion, and multi-LoRA stacking are
planned compatibility work, not current creator controls.

This limitation is the same on 24, 32, 48, and 64 GB profiles; it is a model-format limitation,
not a unified-memory limit. PyTorch node graphs may support standard ComfyUI LoRA workflows, but
that does not make the same LoRA automatically compatible with this MLX generator.

### 3. Make your first clip

1. Install this custom node with ComfyUI Manager, or clone it into `ComfyUI/custom_nodes`.
2. Run the setup command for your memory tier and accept the MiniMax H3 model license.
3. Restart ComfyUI.
4. Load `workflows/minimax_h3_mlx_24gb_turbo4_sol.json` on a 24 GB Mac, or
  `workflows/minimax_h3_mlx_turbo4_sol.json` on larger systems.
5. Change the prompt, keep the first render at 864x480 and five seconds, then queue it.

The workflow saves both MP4 and WAV outputs through standard ComfyUI nodes.

### A prompt that gives the model enough direction

Use one continuous shot and describe picture, motion, consistency, and sound:

```text
A cinematic close-up of a handcrafted ceramic cup on a wooden studio table.
Warm morning light moves slowly across the glaze while the camera makes a gentle orbit.
Keep the cup shape consistent, no cuts, no text.
Audio: quiet room tone, subtle ceramic touch, soft fabric movement, no music.
```

Creator tips:

- Draft at 480p / 5 seconds before committing to a longer render.
- State what must remain consistent: face, product shape, clothes, camera direction, or location.
- Describe sound explicitly. Use `no speech` or `no music` when you do not want them.
- Avoid asking for several cuts in one five-second clip.
- Turbo4 is the reliable preview path; treat Turbo8 as an optional alternate trajectory.

### About the 24 GB preset

The 24 GB workflow uses Qwen8 25+25 staging, Core4 + pruned AdaLN16, two-block uncached offset
streaming, eager Video/Audio VAE stages, and Sol-Attn. It measured a **16.36 GB** process lifetime
peak at 864x480 / 124 frames on the M3 Ultra test host, with media byte-identical to resident output.
It took 239.27 seconds on that host, about 10.35% longer than resident.

This is still an **experimental** tier because it has not been run on a physical 24 GB Mac. Close
other memory-heavy applications, use Turbo4, and start with 480p / 5 seconds. The setup temporarily
needs substantial disk space because it downloads the official BF16-pruned source and builds the
Core4 resident and stream2 layouts locally; allow roughly 110–120 GB free.

## Technical model reference

The node supports quantized creator profiles and BF16 quality profiles. With no `--profile` flag,
setup builds the **32 GB Core4-pruned creator default** locally.

| Profile | Transformer download | Resident after AdaLN drop | Use case |
| --- | ---: | ---: | --- |
| 4-bit core + 8-bit full AdaLN | about 25.3 GB | 11.45 GB | Legacy full-width AdaLN profile |
| 8-bit core + 8-bit full AdaLN | about 35.3 GB | 21.47 GB | Legacy full-width higher-fidelity profile |
| 4-bit core + pruned 16-bit AdaLN | about 11.4 GB | 11.33 GB | 32 GB resident default |
| 8-bit core + pruned 16-bit AdaLN | about 21.4 GB | 21.35 GB | 48 GB resident default |
| 16-bit attention + 8-bit MLP + pruned 16-bit AdaLN | about 29.0 GB | 28.87 GB | 64 GB resident default |
| BF16 core + rank-8 pruned AdaLN | about 40.2 GB | 40.2 GB | 48/64 GB high-quality stream2 tier |
| BF16 core + full-width AdaLN | about 66.3 GB | 40.3 GB | 96 GB+ full-quality tier |

The published 4/8-bit profiles keep AdaLN full-width at 8-bit. Locally built pruned profiles use the official rank-8 curve checkpoint and keep that small AdaLN at 16-bit; padding rank 8 to group size 32 made an 8-bit build slightly larger without reducing resident memory. Comfy checkpoints store QKV rows as `[Q][K][V]`; the MLX repacker converts them to the per-head interleaved layout expected by this runtime.

### Whole-process memory

Transformer resident size is not the amount of unified memory required to finish a generation. The
capacity number to use is the process peak footprint, which also includes MLX allocations, the
PyTorch MPS Sol-Attn bridge, Python, and transient stage allocations. Qwen, DiT, Video VAE, and
Audio VAE run as strict sequential stages, so their individual peaks are not added together.

The following measurements use prequantized Qwen8, 864x480, 124 frames, Turbo4, and Sol-Attn on
the M3 Ultra test system:

| Runtime profile | DiT resident after AdaLN drop | Denoise MLX peak | Process peak RSS | Process peak footprint | Swap growth | Recommended memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Core4 + pruned AdaLN16 resident | 11.33 GB | 15.60 GB | 14.31 GB | **24.70 GB** | 0 GB | 32 GB+ |
| Core8 + pruned AdaLN16 resident | 21.35 GB | 25.62 GB | 22.86 GB | **32.21 GB** | 0 GB | 48 GB+ |
| Attention16 + MLP8 + pruned AdaLN16 resident | 28.87 GB | 33.15 GB | 30.31 GB | **39.73 GB** | 0 GB | 64 GB+ |
| BF16-pruned stream2 | streamed | 7.63 GB | 14.30 GB | **24.70 GB** | 0 GB | 48/64 GB quality mode |
| Full4 resident with full-width AdaLN8 | 11.45 GB | 26.14 GB | 26.60 GB | **35.43 GB** | 0 GB | 48 GB+ |
| Full4 stream5 | 1.48 GB active before release | 6.18 GB | 13.61 GB | **25.77 GB** | 0 GB | 32 GB experimental |

The Core4/Core8 process measurements were captured with the older pruned AdaLN8 checkpoints. They
are conservative proxies for the new AdaLN16 defaults: both variants have the same post-cache DiT
resident size, while the AdaLN16 checkpoint is about 0.10 GB smaller. The Attention16/MLP8/AdaLN16
row is a direct measurement of the current 64 GB profile. Full BF16 resident does not yet have a
complete whole-process peak measurement and is therefore recommended only at 96 GB+.

The setup downloads a pinned six-shard subset of `lmstudio-community/Qwen3-VL-32B-Instruct-MLX-8bit`: embedding plus layers 0-49, exactly the part H3 evaluates. Cross-layer unquantized norm tensors are byte-identical to the MiniMax H3 Qwen checkpoint. The unused final 14 layers, LM head, and seventh shard are not downloaded.

Allow roughly **105 GB free for the default Core4-pruned setup** and **110–120 GB for the 24 GB
bundle** while it builds both resident and stream2 layouts. This includes the 32.13 GB prequantized
Qwen8 subset and the official BF16-pruned source used for local conversion.

## Requirements

- Apple Silicon Mac running macOS
- ComfyUI with Python 3.11+
- About 105 GB free disk space for the default locally built profile
- 32 GB unified memory for the supported creator default; 24 GB remains experimental
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

Download and build the 32 GB creator model set with one command:

```bash
python3 scripts/setup_comfyui.py
```

The command asks you to accept the MiniMax H3 Community License, then installs models under `ComfyUI/models/minimax_h3`. Downloads are resumable through Hugging Face Hub.

Optional profiles:

```bash
# Build the experimental 24 GB Core4 stream2 bundle
python3 scripts/setup_comfyui.py --profile 24

# Install the 8-bit transformer instead of 4-bit
python3 scripts/setup_comfyui.py --profile 8

# Install the BF16-pruned quality profile for stream2
python3 scripts/setup_comfyui.py --profile bf16-pruned

# Build resident pruned quant profiles locally from the shared BF16 source
python3 scripts/setup_comfyui.py --profile 4-pruned
python3 scripts/setup_comfyui.py --profile 8-pruned

# Build the 64 GB resident profile: 16-bit attention, 8-bit MLP, pruned 16-bit AdaLN
python3 scripts/setup_comfyui.py --profile attention16-mlp8

# Install every supported transformer profile
python3 scripts/setup_comfyui.py --profile all

# Non-interactive license confirmation for automation
python3 scripts/setup_comfyui.py --profile 4 --accept-license
```

For unusual ComfyUI layouts, pass `--models-dir /path/to/ComfyUI/models` or set `MINIMAX_H3_MODELS_DIR` before starting ComfyUI.

The pruned quant profiles are never downloaded as third-party converted weights. Setup downloads the official BF16-pruned checkpoint and builds them locally. The Core4/Core8 converter works one block at a time; its measured peak RSS was 3.43 GB, with build times of 65 and 94 seconds on the M3 Ultra test system. The Attention16/MLP8 converter loads the pruned source as one model and is intended for 64 GB+ systems.

## Use

1. Restart ComfyUI after setup.
2. Load `workflows/minimax_h3_mlx_turbo4_sol.json` from this repository.
3. Edit the prompt and queue the workflow.

The workflow saves an MP4 and WAV using ComfyUI's core `CreateVideo`, `SaveVideo`, and `SaveAudio`
nodes. The generic workflow starts with the 32 GB creator configuration:

- DiT: Core4 with pruned AdaLN16, resident
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
| Turbo 8 Balanced | 8 | on | off | on |
| Full 20 Quality | 20 | off | optional, on | on |

Sol-Attn is optional and enabled by default for every preset. Full 20 exposes Safe First Block Cache as a separate option and enables it by default; Turbo modes ignore the FBC setting. Turbo8 is available as a balanced preset but has not yet completed the same perceptual validation as Turbo4.

`memory_mode=auto` selects Core4 stream2 below 30 decimal GB and keeps quantized/mixed profiles
resident above that threshold. `stream_io=auto` selects uncached offset reads for the 24 GB path.
BF16-pruned and full BF16 use stream2 below 96 GB. Prequantized Qwen8 uses 25+25 stages below 40 GB
and a single 50-layer stage otherwise.

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

Pass the resulting directory as `--transformer`. The staged runner detects `streaming_manifest.json` automatically. Streaming supports the same Safe First Block Cache used by Full 20.

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

### Experimental offset I/O

Set `stream_io=offset` in the node, pass `--stream-io offset` on the command line, or set
`MINIMAX_H3_STREAM_IO=offset` to replace core-chunk `mx.load()` calls with direct safetensors
`pread`, best-effort macOS `F_NOCACHE`, and one-chunk lookahead prefetch. The dedicated 24 GB
workflow enables this automatically; other profiles keep standard MLX loading by default.

The minimal Core4 + pruned AdaLN16 prototype used one block per chunk at 864x480, 124 frames,
Turbo4, Qwen8, and Sol-Attn:

The first table comes from the smoke runner's one-second RSS/`top` sampling and is useful for
comparing the resident and offset runs under the same monitor:

| Metric | Resident | Offset stream1 |
| --- | ---: | ---: |
| Denoise MLX peak | 15.60 GB | **5.19 GB** |
| Process RSS peak | 14.38 GB | **13.51 GB** |
| Process footprint peak | **23.62 GB** | 24.70 GB |
| Denoise | **151.62 s** | 188.99 s |
| Wall time | **219.30 s** | 256.61 s |
| Swap growth | 0 GB | 0 GB |

Offset mode read 43.36 GB across 200 block loads. The background reader spent 9.20 seconds in
`pread`, but the foreground waited only 0.15 seconds; prefetch therefore hid almost all physical
I/O. The remaining slowdown came from per-block reconstruction, materialization, garbage
collection, and cache clearing. Resident and offset MP4/WAV outputs were byte-for-byte identical.

Offset I/O alone lowered the DiT MLX peak by 10.41 GB but did not lower the whole-process peak. The
remaining issue was MLX lazy execution in both VAEs: Video VAE accumulated deferred spatial tiles
and temporal clips, while Audio VAE accumulated all seven BigVGAN upsample stages. Evaluating and
clearing each tile, clip, and audio upsample stage immediately keeps the math unchanged while
reusing the live activation workspace.

An exact `proc_pid_rusage` trace sampled every 0.25 seconds after offset I/O and eager execution in
both VAEs measured:

| Stage | Stream1 peak | Stream2 peak |
| --- | ---: | ---: |
| Qwen8 25+25 | 14.74 GB | 14.74 GB |
| Core4 offset-streamed DiT | 11.78 GB | 12.22 GB |
| Video VAE | **15.99 GB** | **16.36 GB** |
| Audio VAE | 7.52 GB | 8.94 GB |
| Process lifetime maximum | **16.06 GB** | **16.36 GB** |

Apple's 24 GiB configuration provides 25.77 decimal GB of physical memory. Stream2 leaves about
9.41 GB above the measured 16.36 GB lifetime maximum and is the better speed/memory balance:
239.27 seconds end to end versus 257.58 seconds stream1 and 216.60 seconds resident. Both optimized
outputs remain byte-for-byte identical to resident output. Video VAE eager execution reduced the
prior 25.76 GB stage peak by 9.77 GB; Audio VAE eager execution reduced its independent decode peak
from 15.24 GB to 4.94 GB without quantizing either VAE.

An isolated 1280x720 / 124-frame Video VAE decode peaked at 16.40 GB with eager tiles, compared to
14.01 GB for 864x480. The complete 720p pipeline has not yet been measured, so 24 GB support remains
an experimental 480p Turbo4 tier.

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
