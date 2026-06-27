# Dapple — Local AI Image Generator

A single-file, interactive terminal tool for generating images with all major **FLUX** models via HuggingFace Diffusers. Works entirely on your local machine — no cloud API, no subscription, no external server.

## Quick Start

```bash
# 1. Install PyTorch with CUDA first:
#    https://pytorch.org/get-started/locally/

# 2. Install dependencies:
pip install -r requirements.txt

# 3. (Optional) Create .env file with your HF token:
#    echo HF_TOKEN=hf_your_token_here > .env

# 4. Run the generator:
python dapple.py
```

## Available Models

| Key | Model | Params | Speed | License | Token? |
|-----|-------|--------|-------|---------|--------|
| `klein-4b` | FLUX.2 Klein 4B | 4B | Fastest | Apache 2.0 | No |
| `klein-9b` | FLUX.2 Klein 9B | 9B | Fast | Non-commercial | Yes |
| `schnell` | FLUX.1 Schnell | 12B | Moderate | Apache 2.0 | No |
| `dev` | FLUX.1 Dev | 12B | Slow | Non-commercial | Yes |
| `flux2-dev` | FLUX.2 Dev | 32B | Slowest | Non-commercial | Yes |

Run `python dapple.py --list` to see full model details.

## How to Use

### Interactive Mode (default)

1. The script asks **"Which model would you like to load?"** — pick by number or key.
2. The model loads (first run downloads ~10–45 GB, cached afterward).
3. Type your **prompt** (required) — describe the image you want.
4. Type a **negative prompt** (optional, press Enter to skip) — describe what to avoid.
5. The image is generated and saved to `Generated_Images_YYYY-MM-DD/`.
6. Repeat! Type `quit`, `q`, or `exit` to stop.

### Configuration Mode

Edit `generator_config.json` to pre-set values. When any config is active, the script:
- Skips model selection (if `model` is set)
- Uses the configured prompt (if `default_prompt` is set)
- Uses the configured negative prompt (if `default_negative_prompt` is set)
- Shows a notice that config mode is active

```json
{
    "model": "klein-4b",
    "default_prompt": "A serene mountain lake at sunset, photorealistic",
    "default_negative_prompt": "blurry, low quality, distorted",
    "cache_dir": "D:/AI_Models",
    "steps": 4,
    "use_nf4": true
}
```

**Leave any field as `""` or `0` to use the interactive flow for that step.**

## HuggingFace Token

Some models require a free HuggingFace token:

1. Accept the license at the model's HF page (linked in the model list)
2. Create a token at https://huggingface.co/settings/tokens
3. Create a `.env` file next to `dapple.py`:
   ```
   HF_TOKEN=hf_your_token_here
   ```

**Models that don't need a token:** `klein-4b`, `schnell`

## Output

Images are saved to a folder named `Generated_Images_YYYY-MM-DD/` (e.g., `Generated_Images_2026-06-27/`). Each filename includes the timestamp and a truncated version of your prompt.

## Configuration Reference (generator_config.json)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `""` | Pre-select model. One of: `schnell`, `dev`, `klein-4b`, `klein-9b`, `flux2-dev` |
| `default_prompt` | string | `""` | Default prompt. If set, prompt step is skipped |
| `default_negative_prompt` | string | `""` | Default negative prompt. If set, negative step is skipped |
| `cache_dir` | string | `""` | Model download location. Empty = `~/.cache/huggingface` |
| `steps` | int | `0` | Inference steps. 0 = use model default |
| `guidance` | float | `0.0` | Guidance scale. 0.0 = use model default |
| `width` | int | `0` | Image width. 0 = use model default |
| `height` | int | `0` | Image height. 0 = use model default |
| `seed` | int | `-1` | Random seed. -1 = random each time |
| `use_nf4` | bool | `true` | 4-bit NF4 quantization (~¼ VRAM) |
| `use_cpu_offload` | string | `"auto"` | `"auto"`, `"always"`, or `"never"` |

## VRAM Requirements (approximate)

| Model | Params | NF4 | Full |
|-------|--------|-----|------|
| FLUX.2 Klein 4B | 4B | ~14 GB | ~22 GB |
| FLUX.2 Klein 9B | 9B | ~17 GB | ~32 GB |
| FLUX.1 Schnell | 12B | ~18 GB | ~38 GB |
| FLUX.1 Dev | 12B | ~18 GB | ~38 GB |
| FLUX.2 Dev | 32B | ~28 GB | ~78 GB |

NF4 is enabled by default (`use_nf4: true`). Models exceeding available VRAM will use CPU offload automatically. Only Klein 4B fits a 16 GB GPU without offloading.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **CUDA out of memory** | Use a smaller model (`klein-4b`), reduce resolution, or enable NF4 in config |
| **401 / gated / token error** | Set `HF_TOKEN` in `.env` and accept the model license on HuggingFace |
| **Module not found** | Run `pip install -r requirements.txt` |
| **Slow generation** | Enable NF4 (`use_nf4: true`), reduce steps, or use `klein-4b` |
| **Model not downloading** | Check internet connection or if original model links have been removed |

## License

This tool itself is provided as-is under the MIT License. Each model has its own license:
- **Apache 2.0**: `klein-4b`, `schnell` — free for commercial use
- **FLUX Non-Commercial**: `dev`, `klein-9b`, `flux2-dev` — research/personal use only

You are responsible for complying with each model's license terms.
