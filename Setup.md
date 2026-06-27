# Dapple — Setup Guide

---

## 1. Configuration File (`generator_config.json`)

`generator_config.json` is the single place to pre-configure the generator. Every field is optional — leave a field empty (or `0`) to use the interactive terminal flow instead.

### Fields

| Field | Type | Default | What it does |
|---|---|---|---|
| `model` | string | `""` | Pre-selects a model. One of: `schnell`, `dev`, `klein-4b`, `klein-9b`, `flux2-dev`. Empty = choose interactively. |
| `default_prompt` | string | `""` | A prompt that is always used. If set, the prompt step is skipped. |
| `default_negative_prompt` | string | `""` | A negative prompt that is always used. If set, the negative prompt step is skipped. |
| `cache_dir` | string | `""` | Where model files are downloaded and stored (10–45 GB per model). The script auto-detects `D:\AI_Models` or `E:\AI_Models` if empty. |
| `steps` | int | `0` | Inference steps. `0` = use the model's recommended default (4 for Schnell/Klein, 20–28 for Dev). |
| `guidance` | float | `0.0` | Guidance scale. `0.0` = use the model's recommended default. Only FLUX.1 Dev responds to values > 1. |
| `width` | int | `0` | Image width in pixels. `0` = model default (1024). |
| `height` | int | `0` | Image height in pixels. `0` = model default (1024). |
| `seed` | int | `-1` | Random seed. `-1` = random each time. Any other integer = reproducible results. |
| `use_nf4` | bool | `true` | 4-bit NF4 quantization. Reduces VRAM usage to ~¼. Turn off only if you have 24 GB+ VRAM. |
| `use_cpu_offload` | string | `"auto"` | `"auto"` = decide per model, `"always"` = force CPU offload (saves VRAM), `"never"` = keep everything on GPU. |

### Usage patterns

**Fully interactive** (no config):
```json
{
    "model": "",
    "default_prompt": "",
    "default_negative_prompt": "",
    "cache_dir": "",
    "steps": 0,
    "guidance": 0.0,
    "width": 0,
    "height": 0,
    "seed": -1,
    "use_nf4": true,
    "use_cpu_offload": "auto"
}
```

**Pre-select model only** (choose at startup, then prompt interactively):
```json
{
    "model": "klein-4b"
}
```

**Fully automated** (one Enter per image):
```json
{
    "model": "klein-4b",
    "default_prompt": "A serene mountain lake at sunset, photorealistic, 8K",
    "default_negative_prompt": "blurry, low quality, distorted"
}
```

### The setup wizard

On first run, if no valid `cache_dir` is found, the script launches a setup wizard that:
- Auto-detects existing model caches on your drives
- Suggests common paths (`D:\AI_Models`, `E:\AI_Models`, HuggingFace default)
- Lets you type a custom path
- Saves your choice to `generator_config.json`

Run `python dapple.py --setup` to re-launch the wizard at any time.

---

## 2. Image Generator (`dapple.py`)

### How it works

A single Python script that loads a FLUX model via HuggingFace Diffusers, accepts text prompts in the terminal, and saves generated images to disk.

**Model lifecycle:**
1. **Select** — pick a model from the numbered menu (or use a pre-configured one).
2. **Load** — the model downloads to `cache_dir` on first run, then reuses cached files. NF4 quantization is applied by default to reduce VRAM.
3. **Generate** — you type a prompt (required), then a negative prompt (optional, Enter to skip). The image is generated and saved.
4. **Repeat or quit** — type `q`, `quit`, or `exit` at any prompt to stop. Ctrl+C interrupts the current generation but keeps the session alive.

**Pipeline classes:**
| Model key | Pipeline class | Diffusers import path |
|---|---|---|
| `schnell` | `FluxPipeline` | `diffusers.pipelines.flux.pipeline_flux` |
| `dev` | `FluxPipeline` | `diffusers.pipelines.flux.pipeline_flux` |
| `klein-4b` | `Flux2KleinPipeline` | `diffusers.pipelines.flux2.pipeline_flux2_klein` |
| `klein-9b` | `Flux2KleinPipeline` | `diffusers.pipelines.flux2.pipeline_flux2_klein` |
| `flux2-dev` | `Flux2Pipeline` | `diffusers.pipelines.flux2.pipeline_flux2` |

**Negative prompts:** Only FLUX.1 Dev (`guidance > 1.0`) responds to negative prompts. FLUX.2 models and FLUX.1 Schnell are distilled — they ignore negative prompts entirely. The script warns you when a model does not support them.

**Output:** Images are saved to `Generated_Images_YYYY-MM-DD/` in the same folder as the script. Each filename includes a timestamp and the first 60 characters of your prompt.

### CLI flags

| Flag | Action |
|---|---|
| *(none)* | Normal interactive mode |
| `--list`, `-l` | List all available models and exit |
| `--setup` | Re-run the cache directory setup wizard |

### Environment

Create a `.env` file next to the script for the HuggingFace token (required for `dev`, `klein-9b`, and `flux2-dev`):

```
HF_TOKEN=hf_your_token_here
```

### Requirements

```
diffusers >= 0.32.0
transformers >= 4.46.0
accelerate >= 1.2.0
bitsandbytes >= 0.45.0
sentencepiece >= 0.2.0
python-dotenv >= 1.0.0
Pillow >= 11.0.0
```

PyTorch with CUDA must be installed separately: https://pytorch.org/get-started/locally/

### VRAM estimates

| Model | Params | NF4 | Full |
|---|---|---|---|
| FLUX.2 Klein 4B | 4B | ~14 GB | ~22 GB |
| FLUX.2 Klein 9B | 9B | ~17 GB | ~32 GB |
| FLUX.1 Schnell | 12B | ~18 GB | ~38 GB |
| FLUX.1 Dev | 12B | ~18 GB | ~38 GB |
| FLUX.2 Dev 32B | 32B | ~28 GB | ~78 GB |

NF4 is on by default (`use_nf4: true`). Models exceeding your VRAM will use CPU offload automatically.
