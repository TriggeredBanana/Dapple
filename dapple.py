#!/usr/bin/env python3
"""Dapple — local AI image generator.

Loads FLUX models via HuggingFace Diffusers, accepts text prompts,
and saves generated PNG images to a dated output folder.

Usage:  python dapple.py [--list] [--setup]
Config: generator_config.json  —  Docs: Setup.md
"""

import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Load HF_TOKEN from .env before torch/diffusers imports
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore


# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

def _load_user_config() -> dict:
    """Read generator_config.json, strip keys prefixed with _comment_."""
    config_path = Path(__file__).parent / "generator_config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"⚠️  Could not parse generator_config.json: {exc}")
        print("   Running without config.\n")
        return {}
    return {k: v for k, v in raw.items() if not k.startswith("_comment")}


def _config_is_active(user_cfg: dict) -> bool:
    """Return True if any workflow field is set (excludes cache_dir)."""
    if not user_cfg:
        return False
    if user_cfg.get("model", "").strip():
        return True
    if user_cfg.get("default_prompt", "").strip():
        return True
    if user_cfg.get("default_negative_prompt", "").strip():
        return True
    if user_cfg.get("steps", 0) > 0:
        return True
    if user_cfg.get("guidance", 0.0) > 0.0:
        return True
    if user_cfg.get("width", 0) > 0:
        return True
    if user_cfg.get("height", 0) > 0:
        return True
    if user_cfg.get("seed", -1) != -1:
        return True
    return False


# ---------------------------------------------------------------------------
#  Model cache
# ---------------------------------------------------------------------------

def _resolve_cache_dir(user_cfg: dict) -> str:
    """Resolve HF_HOME from config, env, or auto-detect; returns "" if none."""
    explicit = (user_cfg.get("cache_dir") or "").strip()
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p.resolve())
        # Config path doesn't exist — warn but still try parent
        if p.parent.exists():
            return str(p.resolve())
        print(f"⚠️  Configured cache_dir '{explicit}' does not exist "
              f"and cannot be created.")

    # ── HF_HOME env variable ─────────────────────────────────────────────────
    env_home = os.environ.get("HF_HOME", "").strip()
    if env_home:
        p = Path(env_home)
        if p.exists():
            return str(p.resolve())

    # ── Auto-detect common model cache drives (Windows) ──────────────────────
    for candidate in (r"D:\AI_Models", r"E:\AI_Models"):
        p = Path(candidate)
        if p.exists():
            return candidate

    # ── No cache found — signal caller to run setup wizard ───────────────────
    return ""


# ---------------------------------------------------------------------------
#  Setup wizard
# ---------------------------------------------------------------------------

def _run_setup_wizard(user_cfg: dict) -> str:
    """Prompt for model storage path, persist to config, return resolved dir."""
    print()
    print("  ╔" + "═" * 64 + "╗")
    print("  ║  FIRST-RUN SETUP — Where should AI models be stored?          ║")
    print("  ║  Models are 10–45 GB each and are reused across runs.         ║")
    print("  ╚" + "═" * 64 + "╝")
    print()

    suggestions: list[str] = []

    # ── Auto-detect existing HuggingFace caches ──────────────────────────────
    for candidate in (r"D:\AI_Models", r"E:\AI_Models",
                      str(Path.home() / ".cache" / "huggingface")):
        p = Path(candidate)
        if p.exists() and any(p.glob("models--*")):
            suggestions.append(candidate)

    # ── Also suggest common drive roots (even if empty) ──────────────────────
    for drive in ("D:/", "E:/", "F:/"):
        p = Path(drive)
        if p.exists():
            suggestion = str(p / "AI_Models")
            if suggestion not in suggestions:
                suggestions.append(suggestion)

    # ── Always include HuggingFace default as a fallback option ──────────────
    hf_default = str(Path.home() / ".cache" / "huggingface")
    if hf_default not in suggestions:
        suggestions.append(hf_default)

    if suggestions:
        print("  Suggested locations:")
        print("  ─────────────────────────────────────────────────────────────")
        for i, s in enumerate(suggestions, 1):
            exists = "✅ (found)" if Path(s).exists() else "📁 (will create)"
            print(f"    [{i}] {s:<50} {exists}")
        print("  ─────────────────────────────────────────────────────────────")
        print()
        print("  Type a number to pick a suggestion, OR type a custom path.")
    else:
        print("  Enter a path where models should be downloaded and stored.")

    print("  Examples:  D:/AI_Models   E:/MyModels   /home/you/models")
    print()

    while True:
        try:
            choice = input("  📁 Model storage path → ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  👋 Setup cancelled. Goodbye!\n")
            sys.exit(0)

        if not choice:
            print("  ❌ Path cannot be empty.\n")
            continue

        # ── Number = pick from suggestions ───────────────────────────────────
        if choice.isdigit() and suggestions:
            idx = int(choice)
            if 1 <= idx <= len(suggestions):
                choice = suggestions[idx - 1]

        # ── Validate and create ──────────────────────────────────────────────
        p = Path(choice.strip())
        try:
            p.mkdir(parents=True, exist_ok=True)
            resolved = str(p.resolve())
            print(f"\n  ✅ Models will be stored at: {resolved}")

            # Persist to config
            _save_config_key("cache_dir", choice.strip())
            user_cfg["cache_dir"] = choice.strip()

            return resolved
        except OSError as exc:
            print(f"  ❌ Cannot create '{choice}': {exc}")
            print("  Try a different path.\n")


def _save_config_key(key: str, value) -> None:
    """Write a single key to generator_config.json, preserving existing keys."""
    config_path = Path(__file__).parent / "generator_config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        data = {}
    data[key] = value
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)
        fh.write("\n")


_cache_dir: str = ""  # set during startup after config is available


def _setup_hardware():
    """Enable TF32 matmuls and high-precision float32."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
#  Model definitions
# ---------------------------------------------------------------------------

# Keys: name, repo_id, pipeline_module, license, requires_token,
#       default_steps, default_guidance, default_width, default_height,
#       max_sequence_length, supports_negative, needs_cpu_offload,
#       nf4_recommended, description
MODELS: dict = {
    "schnell": {
        "name":                 "FLUX.1 Schnell",
        "repo_id":              "black-forest-labs/FLUX.1-schnell",
        "pipeline_module":      "FluxPipeline",
        "license":              "Apache 2.0 — free for commercial use",
        "requires_token":       False,
        "default_steps":        4,
        "default_guidance":     0.0,
        "default_width":        1024,
        "default_height":       1024,
        "max_sequence_length":  256,
        "supports_negative":    False,
        "needs_cpu_offload":    "auto",
        "nf4_recommended":      True,
        "description":          "12B · Moderate · Apache 2.0 · no token needed",
    },
    "dev": {
        "name":                 "FLUX.1 Dev",
        "repo_id":              "black-forest-labs/FLUX.1-dev",
        "pipeline_module":      "FluxPipeline",
        "license":              "Non-commercial only — requires HF token",
        "requires_token":       True,
        "default_steps":        20,
        "default_guidance":     3.5,
        "default_width":        1024,
        "default_height":       1024,
        "max_sequence_length":  512,
        "supports_negative":    True,
        "needs_cpu_offload":    "always",
        "nf4_recommended":      True,
        "description":          "12B · Slow · non-commercial · needs HF token",
    },
    "klein-4b": {
        "name":                 "FLUX.2 Klein 4B",
        "repo_id":              "black-forest-labs/FLUX.2-klein-4B",
        "pipeline_module":      "Flux2KleinPipeline",
        "license":              "Apache 2.0 — free for commercial use",
        "requires_token":       False,
        "default_steps":        4,
        "default_guidance":     1.0,
        "default_width":        1024,
        "default_height":       1024,
        "max_sequence_length":  512,
        "supports_negative":    False,
        "needs_cpu_offload":    "auto",
        "nf4_recommended":      True,
        "description":          "4B · Fastest · Apache 2.0 · no token needed",
    },
    "klein-9b": {
        "name":                 "FLUX.2 Klein 9B",
        "repo_id":              "black-forest-labs/FLUX.2-klein-9B",
        "pipeline_module":      "Flux2KleinPipeline",
        "license":              "Non-commercial only — requires HF token",
        "requires_token":       True,
        "default_steps":        4,
        "default_guidance":     1.0,
        "default_width":        1024,
        "default_height":       1024,
        "max_sequence_length":  512,
        "supports_negative":    False,
        "needs_cpu_offload":    "always",
        "nf4_recommended":      True,
        "description":          "9B · Fast · non-commercial · needs HF token",
    },
    "flux2-dev": {
        "name":                 "FLUX.2 Dev 32B",
        "repo_id":              "black-forest-labs/FLUX.2-dev",
        "pipeline_module":      "Flux2Pipeline",
        "license":              "Non-commercial only — requires HF token",
        "requires_token":       True,
        "default_steps":        28,
        "default_guidance":     4.0,
        "default_width":        1024,
        "default_height":       1024,
        "max_sequence_length":  512,
        "supports_negative":    False,
        "needs_cpu_offload":    "always",
        "nf4_recommended":      True,
        "description":          "32B · Slowest · non-commercial · needs HF token",
    },
}

VALID_MODEL_KEYS = list(MODELS.keys())


# ---------------------------------------------------------------------------
#  Pipeline (singleton — loaded once, reused until quit)
# ---------------------------------------------------------------------------

_pipeline = None
_active_model_key: str = ""


def _get_pipeline() -> object:
    """Return cached pipeline or None."""
    return _pipeline


def _pipeline_is_loaded() -> bool:
    return _pipeline is not None and bool(_active_model_key)


def load_model(model_key: str, user_cfg: dict) -> object:
    """Load and cache a diffusers pipeline. Handles token, NF4, offload, VAE."""
    global _pipeline, _active_model_key

    # Return cached pipeline if model unchanged
    if _pipeline is not None and _active_model_key == model_key:
        return _pipeline

    if _pipeline is not None and _active_model_key != model_key:
        print(f"\n🔄 Switching from {_active_model_key} to {model_key} ...")
        _unload_pipeline()

    cfg = MODELS[model_key]
    hf_token = os.environ.get("HF_TOKEN", "").strip()

    # Token gate
    if cfg["requires_token"] and not hf_token:
        print(f"\n{'─' * 72}")
        print(f"  ⚠️  '{model_key}' ({cfg['name']}) requires a HuggingFace token.")
        print(f"  How to fix:")
        print(f"    1. Accept the license at:")
        print(f"       https://huggingface.co/{cfg['repo_id']}")
        print(f"    2. Create a token at:")
        print(f"       https://huggingface.co/settings/tokens")
        print(f"    3. Create a .env file next to this script with:")
        print(f"       HF_TOKEN=hf_your_token_here")
        print(f"{'─' * 72}\n")
        sys.exit(1)

    print(f"\n{'─' * 72}")
    print(f"  📦 Loading {cfg['name']}")
    print(f"  📂 Repo:    {cfg['repo_id']}")
    print(f"  💾 Cache:   {_cache_dir}")
    print(f"  📐 Default: {cfg['default_width']}×{cfg['default_height']}, "
          f"{cfg['default_steps']} steps, guidance={cfg['default_guidance']}")
    print(f"  📜 License: {cfg['license']}")

    load_kwargs: dict = {
        "torch_dtype": _get_torch_dtype(),
        "cache_dir": _cache_dir,
    }
    if hf_token:
        load_kwargs["token"] = hf_token

    # NF4 quantization
    use_nf4 = _resolve_nf4(user_cfg, model_key)
    if use_nf4:
        try:
            from diffusers.quantizers.pipe_quant_config import \
                PipelineQuantizationConfig
            load_kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": "bfloat16",
                },
                components_to_quantize=["transformer"],
            )
            print("  ⚡ NF4 quantization: ENABLED (~¼ VRAM usage)")
        except Exception as exc:
            print(f"  ⚠️  NF4 unavailable ({exc}) — using full precision")

    print(f"  ⏳ Downloading/loading model (first run downloads ~10-45 GB)...")
    try:
        pipe = _load_pipe_instance(cfg["pipeline_module"], cfg["repo_id"], load_kwargs)
    except Exception as exc:
        _print_load_error(exc, cfg)
        sys.exit(1)

    # CPU offload vs full GPU
    offload_mode = _resolve_cpu_offload(user_cfg, model_key)
    if offload_mode == "always":
        pipe.enable_model_cpu_offload()
        print("  🖥️  CPU offload: ACTIVE")
    elif offload_mode == "auto":
        try:
            pipe = pipe.to("cuda")
            print("  🖥️  GPU: full model on CUDA")
        except Exception:
            pipe.enable_model_cpu_offload()
            print("  🖥️  CPU offload: ACTIVE (GPU OOM fallback)")
    else:
        pipe = pipe.to("cuda")
        print("  🖥️  GPU: full model on CUDA (offload disabled)")

    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    _pipeline = pipe
    _active_model_key = model_key
    print(f"  ✅ {cfg['name']} loaded and ready.")
    print(f"{'─' * 72}\n")
    return _pipeline


def _get_torch_dtype():
    """Return bfloat16 dtype."""
    import torch
    return torch.bfloat16


def _resolve_nf4(user_cfg: dict, model_key: str) -> bool:
    """Config override or model recommendation for NF4."""
    if "use_nf4" in user_cfg:
        return bool(user_cfg["use_nf4"])
    return MODELS[model_key]["nf4_recommended"]


def _resolve_cpu_offload(user_cfg: dict, model_key: str) -> str:
    """Resolve CPU offload: 'always', 'auto', or 'never'."""
    cfg_val = (user_cfg.get("use_cpu_offload") or "auto").strip().lower()
    if cfg_val in ("always", "on", "true", "yes"):
        return "always"
    if cfg_val in ("never", "off", "false", "no"):
        return "never"
    return MODELS[model_key]["needs_cpu_offload"]


def _load_pipe_instance(pipeline_module: str, repo_id: str, load_kwargs: dict):
    """Instantiate the correct diffusers pipeline class."""
    if pipeline_module == "FluxPipeline":
        from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
        return FluxPipeline.from_pretrained(repo_id, **load_kwargs)
    elif pipeline_module == "Flux2KleinPipeline":
        from diffusers.pipelines.flux2.pipeline_flux2_klein import \
            Flux2KleinPipeline
        return Flux2KleinPipeline.from_pretrained(repo_id, **load_kwargs)
    elif pipeline_module == "Flux2Pipeline":
        from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
        return Flux2Pipeline.from_pretrained(repo_id, **load_kwargs)
    else:
        raise ValueError(f"Unknown pipeline module: {pipeline_module}")


def _unload_pipeline():
    """Delete pipeline and empty CUDA cache."""
    global _pipeline, _active_model_key
    if _pipeline is not None:
        del _pipeline
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _pipeline = None
    _active_model_key = ""


def _print_load_error(error: Exception, cfg: dict):
    """Print diagnostic for common pipeline load failures."""
    err_str = str(error).lower()
    print(f"\n  ❌ Model load failed: {error}\n")

    if any(x in err_str for x in ["401", "403", "gated", "token",
                                   "forbidden", "access", "authorization"]):
        print("  🔑 Authentication / access error.")
        print(f"     → Accept the license: https://huggingface.co/{cfg['repo_id']}")
        print(f"     → Create a token:     https://huggingface.co/settings/tokens")
        print(f"     → Set in .env file:   HF_TOKEN=hf_your_token_here")
    elif any(x in err_str for x in ["out of memory", "oom"]):
        print("  💾 Out of GPU VRAM.")
        print("     → Try a smaller model (e.g. 'klein-4b' or 'schnell').")
        print("     → Reduce width/height in generator_config.json.")
        print("     → Make sure 'use_nf4' is set to true in the config.")
        print("     → Close other GPU programs.")
    elif "no module named" in err_str:
        print("  📦 Missing Python package.")
        print("     → Run: pip install -r requirements.txt")
    elif any(x in err_str for x in ["not found", "does not exist",
                                     "connection", "timeout"]):
        print("  🌐 Network / repo error.")
        print("     → Check your internet connection.")
        print(f"     → Verify the repo: https://huggingface.co/{cfg['repo_id']}")


# ---------------------------------------------------------------------------
#  Image generation
# ---------------------------------------------------------------------------

def generate_image(prompt: str, negative_prompt: str, user_cfg: dict) -> Path:
    """Run inference with the loaded pipeline, save PNG, return path."""
    model_key = _active_model_key
    model_cfg = MODELS[model_key]
    pipe = _get_pipeline()

    # Output directory
    today_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(f"Generated_Images_{today_str}")
    output_dir.mkdir(exist_ok=True)

    # Generation parameters: config overrides model defaults
    steps = user_cfg.get("steps", 0) or model_cfg["default_steps"]
    cfg_guidance = user_cfg.get("guidance")
    if cfg_guidance is not None and cfg_guidance != 0.0:
        guidance = float(cfg_guidance)
    else:
        guidance = model_cfg["default_guidance"]
    width = user_cfg.get("width", 0) or model_cfg["default_width"]
    height = user_cfg.get("height", 0) or model_cfg["default_height"]
    seed_val = user_cfg.get("seed", -1)

    import torch
    if seed_val == -1:
        seed_val = int(torch.randint(0, 2 ** 32, (1,)).item())
    generator = torch.Generator(device="cpu").manual_seed(seed_val)

    # Common kwargs
    gen_kwargs: dict = {
        "prompt":           prompt,
        "num_inference_steps": steps,
        "guidance_scale":   guidance,
        "width":            width,
        "height":           height,
        "generator":        generator,
    }

    # Pipeline-specific kwargs
    if model_cfg["pipeline_module"] == "FluxPipeline":
        gen_kwargs["max_sequence_length"] = model_cfg["max_sequence_length"]
        if negative_prompt.strip() and model_cfg["supports_negative"]:
            gen_kwargs["negative_prompt"] = negative_prompt
            gen_kwargs["negative_prompt_2"] = negative_prompt
    elif model_cfg["pipeline_module"] in ("Flux2KleinPipeline", "Flux2Pipeline"):
        gen_kwargs["max_sequence_length"] = model_cfg["max_sequence_length"]

    # Inference
    print(f"  🎨 Generating... ({width}×{height}, {steps} steps, "
          f"guidance={guidance}, seed={seed_val})")
    gen_start = time.time()
    result = pipe(**gen_kwargs)  # type: ignore[operator]
    gen_time = time.time() - gen_start

    # Save
    image = result.images[0]  # type: ignore[union-attr]
    safe_prompt = _sanitize_for_filename(prompt[:60])
    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"{timestamp}_{safe_prompt}.png"
    output_path = output_dir / filename
    image.save(str(output_path), "PNG")
    print(f"  💾 Saved: {output_path.name}  ({gen_time:.1f}s, seed={seed_val})")
    return output_path


def _sanitize_for_filename(text: str) -> str:
    """Strip characters unsafe in filenames."""
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_ ")[:120]


# ---------------------------------------------------------------------------
#  Terminal UI
# ---------------------------------------------------------------------------

def print_banner():
    """Print startup banner."""
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + "  Dapple — Local AI Image Generator".center(70) + "║")
    print("║" + "  Type 'quit' or 'q' at any prompt to exit.".center(70) + "║")
    print("╚" + "═" * 70 + "╝")
    print()


def _print_config_notice(user_cfg: dict):
    """Print active config overrides if any."""
    active_parts = []
    if user_cfg.get("model", "").strip():
        active_parts.append(f"model={user_cfg['model']}")
    if user_cfg.get("default_prompt", "").strip():
        active_parts.append("default prompt set")
    if user_cfg.get("default_negative_prompt", "").strip():
        active_parts.append("default negative prompt set")
    if user_cfg.get("steps", 0) > 0:
        active_parts.append(f"steps={user_cfg['steps']}")
    if user_cfg.get("guidance", 0.0) > 0.0:
        active_parts.append(f"guidance={user_cfg['guidance']}")
    if user_cfg.get("width", 0) > 0:
        active_parts.append(f"size={user_cfg['width']}×{user_cfg.get('height', 0)}")
    if user_cfg.get("seed", -1) != -1:
        active_parts.append(f"seed={user_cfg['seed']}")

    if active_parts:
        print(f"  ⚙️  Config active: {', '.join(active_parts)}")
        print(f"  💡 Edit generator_config.json to change these.\n")


def select_model(user_cfg: dict) -> str:
    """Return model key from config or interactive menu."""
    pre_selected = (user_cfg.get("model") or "").strip().lower()
    if pre_selected:
        if pre_selected in MODELS:
            return pre_selected
        else:
            print(f"  ⚠️  Config model '{pre_selected}' is not valid.")
            print(f"  Valid options: {', '.join(VALID_MODEL_KEYS)}")
            print(f"  Falling back to interactive selection.\n")

    while True:
        print("  Available models:")
        print("  ─────────────────────────────────────────────────────")
        for i, key in enumerate(VALID_MODEL_KEYS, 1):
            m = MODELS[key]
            print(f"    [{i}] {key:<14} {m['description']}")
        print("  ─────────────────────────────────────────────────────")
        print()
        try:
            choice = input("  Enter model key or number → ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  👋 Goodbye!\n")
            sys.exit(0)

        if choice in ("q", "quit", "exit"):
            print("\n  👋 Goodbye!\n")
            sys.exit(0)

        # Try matching by key name
        if choice in MODELS:
            return choice

        # Try matching by number
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(VALID_MODEL_KEYS):
                return VALID_MODEL_KEYS[idx - 1]

        print(f"  ❌ '{choice}' is not a valid choice. "
              f"Pick 1-{len(VALID_MODEL_KEYS)} or a model key.\n")


def get_prompt(user_cfg: dict) -> str:
    """Return config prompt or prompt user; loops on empty input."""
    pre_set = (user_cfg.get("default_prompt") or "").strip()
    if pre_set:
        print(f"  📝 Using configured prompt: \"{pre_set[:80]}{'...' if len(pre_set) > 80 else ''}\"\n")
        return pre_set

    print()
    print("  ── PROMPT ───────────────────────────────────────────")
    print("  Describe the image you want to generate.")
    print("  Be specific — include subject, style, lighting, mood.")
    print()
    prompt = input("  ✏️  Prompt → ").strip()

    if prompt.lower() in ("q", "quit", "exit"):
        print("\n  👋 Goodbye!\n")
        sys.exit(0)

    if not prompt:
        print("  ❌ Prompt cannot be empty. Please try again.")
        return get_prompt(user_cfg)

    return prompt


def get_negative_prompt(user_cfg: dict, model_key: str) -> str:
    """Return config negative prompt or prompt user; Enter to skip."""
    model_cfg = MODELS[model_key]

    pre_set = (user_cfg.get("default_negative_prompt") or "").strip()
    if pre_set:
        if not model_cfg["supports_negative"]:
            print(f"  ⚠️  Note: {model_cfg['name']} does NOT support negative "
                  f"prompts (distilled model).\n")
        else:
            print(f"  🚫 Using configured negative prompt: "
                  f"\"{pre_set[:80]}{'...' if len(pre_set) > 80 else ''}\"\n")
        return pre_set

    print()
    print("  ── NEGATIVE PROMPT (optional) ───────────────────────")
    print("  Describe what you do NOT want in the image.")
    print("  Press Enter to skip (no negative prompt).")
    if not model_cfg["supports_negative"]:
        print(f"  ⚠️  Note: {model_cfg['name']} is a distilled model — "
              f"negative prompts have NO effect.")
    print()

    neg = input("  🚫 Negative prompt → ").strip()

    if neg.lower() in ("q", "quit", "exit"):
        print("\n  👋 Goodbye!\n")
        sys.exit(0)

    return neg


# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------

def interactive_loop(model_key: str, user_cfg: dict):
    """Prompt → generate → repeat until quit/EOF/Ctrl+C."""
    model_cfg = MODELS[model_key]

    # Config-driven mode: both prompt and negative are pre-set
    prompt_pre_set = bool((user_cfg.get("default_prompt") or "").strip())
    neg_pre_set = bool((user_cfg.get("default_negative_prompt") or "").strip())
    config_driven = prompt_pre_set and neg_pre_set

    print(f"\n  🟢 Model: {model_cfg['name']} — ready for prompts.")
    print(f"  📁 Images save to: Generated_Images_{datetime.now().strftime('%Y-%m-%d')}/")
    if config_driven:
        print(f"  ⚙️  Config-driven mode — prompts are pre-set. Press Enter to generate again, or type 'q' to quit.")
    else:
        print(f"  💡 Type 'quit', 'q', or 'exit' at any prompt to stop.")
    print()

    _prompt_notice_shown = False
    _neg_notice_shown = False

    while True:
        try:
            # Prompt
            if prompt_pre_set and _prompt_notice_shown:
                # Suppress repeated notice; just get the pre-set prompt
                prompt = (user_cfg.get("default_prompt") or "").strip()
            else:
                prompt = get_prompt(user_cfg)
                if prompt_pre_set:
                    _prompt_notice_shown = True

            # Config-driven mode: generate, then [Enter] or [q]
            if config_driven:
                print(f"  📝 Prompt: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"")
                neg = (user_cfg.get("default_negative_prompt") or "").strip()
                if neg:
                    print(f"  🚫 Negative: \"{neg[:80]}{'...' if len(neg) > 80 else ''}\"")
                print()
                output_path = generate_image(prompt, neg, user_cfg)
                print(f"  ✅ Done! → {output_path}")
                print()
                again = input("  [Enter] Generate again  |  [q] Quit → ").strip().lower()
                if again in ("q", "quit", "exit"):
                    print("\n  👋 Goodbye!\n")
                    sys.exit(0)
                print()
                continue

            # Negative prompt (interactive mode)
            if neg_pre_set and _neg_notice_shown:
                negative_prompt = (user_cfg.get("default_negative_prompt") or "").strip()
            else:
                negative_prompt = get_negative_prompt(user_cfg, model_key)
                if neg_pre_set:
                    _neg_notice_shown = True

            # Generate
            print()
            output_path = generate_image(prompt, negative_prompt, user_cfg)
            print(f"  ✅ Done! → {output_path}\n")

        except KeyboardInterrupt:
            print("\n\n  ⏹️  Interrupted (Ctrl+C).")
            print("  Type 'quit' to exit or continue with a new prompt.\n")
            continue
        except EOFError:
            print("\n\n  👋 Input closed — goodbye!\n")
            sys.exit(0)
        except Exception as exc:
            print(f"\n  ❌ Generation error: {exc}")
            traceback.print_exc()
            print("  Continuing — you can try again.\n")
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main():
    global _cache_dir

    if "--list" in sys.argv or "-l" in sys.argv:
        _print_model_list()
        return

    user_cfg = _load_user_config()

    _cache_dir = _resolve_cache_dir(user_cfg)

    # First-run or --setup: run setup wizard
    if "--setup" in sys.argv or not _cache_dir:
        if not _cache_dir:
            print("\n  ⚠️  No valid model cache directory found.")
            print("  Running first-run setup to choose where models go...")
        _cache_dir = _run_setup_wizard(user_cfg)
    os.environ["HF_HOME"] = _cache_dir

    _setup_hardware()

    import torch
    if not torch.cuda.is_available():
        print("\n  ❌ CUDA is not available. This tool requires an NVIDIA GPU.")
        print("  Make sure you have PyTorch with CUDA installed.")
        print("  Visit: https://pytorch.org/get-started/locally/\n")
        sys.exit(1)

    print_banner()
    print(f"  🖥️  GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM)")
    print(f"  💾 Cache: {_cache_dir}")

    if _config_is_active(user_cfg):
        _print_config_notice(user_cfg)
    else:
        print(f"  ⚙️  No config overrides active — running interactively.")
        print(f"  💡 To pre-configure, edit generator_config.json\n")

    # Model selection
    try:
        model_key = select_model(user_cfg)
    except (EOFError, KeyboardInterrupt):
        print("\n  👋 Goodbye!\n")
        return

    # Load pipeline
    try:
        load_model(model_key, user_cfg)
    except KeyboardInterrupt:
        print("\n\n  ⏹️  Model loading interrupted.\n")
        return
    except Exception as exc:
        print(f"\n  ❌ Failed to load model: {exc}\n")
        sys.exit(1)

    interactive_loop(model_key, user_cfg)


def _print_model_list():
    """Print model table and exit."""
    print()
    print("  Available models:")
    print("  ─────────────────────────────────────────────────────────────────")
    for key in VALID_MODEL_KEYS:
        m = MODELS[key]
        print(f"  {key:<14} {m['description']}")
        print(f"              Repo:    {m['repo_id']}")
        print(f"              License: {m['license']}")
        print(f"              Default: {m['default_width']}×{m['default_height']}, "
              f"{m['default_steps']} steps, guidance={m['default_guidance']}")
        if m["requires_token"]:
            print(f"              ⚠️  Requires HuggingFace token (.env: HF_TOKEN=...)")
        print()
    print("  Usage: python dapple.py")
    print()


if __name__ == "__main__":
    main()
