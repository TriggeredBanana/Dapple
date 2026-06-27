#!/usr/bin/env python3
"""Dapple Web — local AI image generator with a browser interface.

FastAPI backend wrapping the Dapple image generation pipeline.
Start with: python server.py [--port 8000] [--host 0.0.0.0]
"""

import asyncio
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
#  Config helpers (mirrors dapple.py for independence)
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "generator_config.json"

MODELS = {
    "schnell": {
        "name": "FLUX.1 Schnell",
        "repo_id": "black-forest-labs/FLUX.1-schnell",
        "pipeline_module": "FluxPipeline",
        "license": "Apache 2.0 — free for commercial use",
        "requires_token": False,
        "default_steps": 4,
        "default_guidance": 0.0,
        "default_width": 1024,
        "default_height": 1024,
        "max_sequence_length": 256,
        "supports_negative": False,
        "needs_cpu_offload": "auto",
        "nf4_recommended": True,
        "description": "12B · Moderate · Apache 2.0 · no token needed",
    },
    "dev": {
        "name": "FLUX.1 Dev",
        "repo_id": "black-forest-labs/FLUX.1-dev",
        "pipeline_module": "FluxPipeline",
        "license": "Non-commercial only — requires HF token",
        "requires_token": True,
        "default_steps": 20,
        "default_guidance": 3.5,
        "default_width": 1024,
        "default_height": 1024,
        "max_sequence_length": 512,
        "supports_negative": True,
        "needs_cpu_offload": "always",
        "nf4_recommended": True,
        "description": "12B · Slow · non-commercial · needs HF token",
    },
    "klein-4b": {
        "name": "FLUX.2 Klein 4B",
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "pipeline_module": "Flux2KleinPipeline",
        "license": "Apache 2.0 — free for commercial use",
        "requires_token": False,
        "default_steps": 4,
        "default_guidance": 1.0,
        "default_width": 1024,
        "default_height": 1024,
        "max_sequence_length": 512,
        "supports_negative": False,
        "needs_cpu_offload": "auto",
        "nf4_recommended": True,
        "description": "4B · Fastest · Apache 2.0 · no token needed",
    },
    "klein-9b": {
        "name": "FLUX.2 Klein 9B",
        "repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "pipeline_module": "Flux2KleinPipeline",
        "license": "Non-commercial only — requires HF token",
        "requires_token": True,
        "default_steps": 4,
        "default_guidance": 1.0,
        "default_width": 1024,
        "default_height": 1024,
        "max_sequence_length": 512,
        "supports_negative": False,
        "needs_cpu_offload": "always",
        "nf4_recommended": True,
        "description": "9B · Fast · non-commercial · needs HF token",
    },
    "flux2-dev": {
        "name": "FLUX.2 Dev 32B",
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "pipeline_module": "Flux2Pipeline",
        "license": "Non-commercial only — requires HF token",
        "requires_token": True,
        "default_steps": 28,
        "default_guidance": 4.0,
        "default_width": 1024,
        "default_height": 1024,
        "max_sequence_length": 512,
        "supports_negative": False,
        "needs_cpu_offload": "always",
        "nf4_recommended": True,
        "description": "32B · Slowest · non-commercial · needs HF token",
    },
}

VALID_MODEL_KEYS = list(MODELS.keys())


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}
    return {k: v for k, v in raw.items() if not k.startswith("_") and k != "//"}


def _save_config(data: dict) -> None:
    existing = _load_config() if CONFIG_PATH.exists() else {}
    existing.update(data)
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=4, ensure_ascii=False)
        fh.write("\n")


def _resolve_cache_dir() -> str:
    cfg = _load_config()
    explicit = (cfg.get("cache_dir") or "").strip()
    if explicit and Path(explicit).parent.exists():
        return str(Path(explicit).resolve())

    env_home = os.environ.get("HF_HOME", "").strip()
    if env_home and Path(env_home).exists():
        return env_home

    for candidate in (r"D:\AI_Models", r"E:\AI_Models"):
        if Path(candidate).exists():
            return candidate

    return ""


# ---------------------------------------------------------------------------
#  Pipeline management
# ---------------------------------------------------------------------------

_pipeline = None
_active_model_key: str = ""
_cache_dir: str = ""


def _get_torch_dtype():
    import torch
    return torch.bfloat16


def _setup_hardware():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def load_model(model_key: str) -> dict:
    global _pipeline, _active_model_key

    if _pipeline is not None and _active_model_key == model_key:
        return {"status": "already_loaded", "model": model_key}

    if _pipeline is not None and _active_model_key != model_key:
        _unload_pipeline()

    cfg = MODELS[model_key]
    hf_token = os.environ.get("HF_TOKEN", "").strip()

    if cfg["requires_token"] and not hf_token:
        raise HTTPException(
            status_code=401,
            detail=f"Model '{model_key}' requires a HuggingFace token. "
                   f"Accept the license at https://huggingface.co/{cfg['repo_id']} "
                   f"and set HF_TOKEN in your .env file.",
        )

    user_cfg = _load_config()
    load_kwargs = {
        "torch_dtype": _get_torch_dtype(),
        "cache_dir": _cache_dir,
    }
    if hf_token:
        load_kwargs["token"] = hf_token

    use_nf4 = user_cfg.get("use_nf4", True)
    if use_nf4:
        try:
            from diffusers.quantizers.pipe_quant_config import PipelineQuantizationConfig
            load_kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": "bfloat16",
                },
                components_to_quantize=["transformer"],
            )
        except Exception:
            pass

    if cfg["pipeline_module"] == "FluxPipeline":
        from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
        pipe = FluxPipeline.from_pretrained(cfg["repo_id"], **load_kwargs)
    elif cfg["pipeline_module"] == "Flux2KleinPipeline":
        from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(cfg["repo_id"], **load_kwargs)
    elif cfg["pipeline_module"] == "Flux2Pipeline":
        from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
        pipe = Flux2Pipeline.from_pretrained(cfg["repo_id"], **load_kwargs)
    else:
        raise ValueError(f"Unknown pipeline: {cfg['pipeline_module']}")

    offload_mode = (user_cfg.get("use_cpu_offload") or cfg["needs_cpu_offload"]).strip().lower()
    if offload_mode in ("always", "on", "true", "yes"):
        pipe.enable_model_cpu_offload()
    elif offload_mode in ("never", "off", "false", "no"):
        pipe = pipe.to("cuda")
    else:
        try:
            pipe = pipe.to("cuda")
        except Exception:
            pipe.enable_model_cpu_offload()

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
    return {"status": "loaded", "model": model_key, "name": cfg["name"]}


def _unload_pipeline():
    global _pipeline, _active_model_key
    if _pipeline is not None:
        del _pipeline
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _pipeline = None
    _active_model_key = ""


def _sanitize_filename(text: str) -> str:
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_ ")[:120]


def generate_image(prompt: str, negative_prompt: str = "", seed_val: int = -1) -> dict:
    if _pipeline is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")

    model_cfg = MODELS[_active_model_key]
    user_cfg = _load_config()

    steps = user_cfg.get("steps", 0) or model_cfg["default_steps"]
    width = user_cfg.get("width", 0) or model_cfg["default_width"]
    height = user_cfg.get("height", 0) or model_cfg["default_height"]
    guidance = user_cfg.get("guidance", 0.0) or model_cfg["default_guidance"]

    import torch
    if seed_val == -1:
        seed_val = int(torch.randint(0, 2 ** 32, (1,)).item())

    generator = torch.Generator(device="cpu").manual_seed(seed_val)
    gen_kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height,
        "generator": generator,
    }

    if model_cfg["pipeline_module"] == "FluxPipeline":
        gen_kwargs["max_sequence_length"] = model_cfg["max_sequence_length"]
        if negative_prompt.strip() and model_cfg["supports_negative"]:
            gen_kwargs["negative_prompt"] = negative_prompt
            gen_kwargs["negative_prompt_2"] = negative_prompt
    elif model_cfg["pipeline_module"] in ("Flux2KleinPipeline", "Flux2Pipeline"):
        gen_kwargs["max_sequence_length"] = model_cfg["max_sequence_length"]

    today_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(f"Generated_Images_{today_str}")
    output_dir.mkdir(exist_ok=True)

    gen_start = time.time()
    result = _pipeline(**gen_kwargs)
    gen_time = time.time() - gen_start

    image = result.images[0]
    safe_prompt = _sanitize_filename(prompt[:60])
    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"{timestamp}_{safe_prompt}.png"
    output_path = output_dir / filename
    image.save(str(output_path), "PNG")

    return {
        "filename": filename,
        "path": str(output_path),
        "time": round(gen_time, 1),
        "seed": seed_val,
        "width": width,
        "height": height,
    }


async def event_generator():
    steps = 28
    for i in range(steps + 1):
        yield f"data: {json.dumps({'step': i, 'total': steps})}\n\n"
        await asyncio.sleep(0.05)
    yield "data: {\"done\": true}\n\n"


# ---------------------------------------------------------------------------
#  FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Dapple", version="1.0.0", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
#  Pydantic models
# ---------------------------------------------------------------------------

class ModelSelect(BaseModel):
    model_key: str

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""

class ConfigUpdate(BaseModel):
    key: str
    value: str

# ---------------------------------------------------------------------------
#  API Routes
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def api_status():
    import torch
    gpu_info = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "vram_gb": round(props.total_memory / 1024**3, 1),
        }

    active_model = None
    if _active_model_key:
        active_model = {
            "key": _active_model_key,
            "name": MODELS[_active_model_key]["name"],
        }

    return {
        "gpu": gpu_info,
        "cuda_available": torch.cuda.is_available(),
        "cache_dir": _cache_dir,
        "active_model": active_model,
        "config": _load_config(),
    }


@app.get("/api/models")
async def api_models():
    return {
        "models": [
            {
                "key": key,
                "name": m["name"],
                "description": m["description"],
                "license": m["license"],
                "requires_token": m["requires_token"],
                "default_steps": m["default_steps"],
                "default_guidance": m["default_guidance"],
                "default_width": m["default_width"],
                "default_height": m["default_height"],
                "supports_negative": m["supports_negative"],
            }
            for key, m in MODELS.items()
        ],
    }


@app.post("/api/load-model")
async def api_load_model(req: ModelSelect):
    if req.model_key not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model_key}")
    try:
        result = load_model(req.model_key)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    try:
        result = generate_image(req.prompt, req.negative_prompt)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/config")
async def api_get_config():
    return _load_config()


@app.post("/api/config")
async def api_set_config(req: ConfigUpdate):
    cfg = _load_config()
    try:
        parsed = json.loads(req.value)
    except (json.JSONDecodeError, TypeError):
        parsed = req.value
    cfg[req.key] = parsed
    _save_config(cfg)
    return {"status": "ok", "key": req.key, "value": parsed}


@app.get("/api/progress")
async def api_progress():
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/image/{filename}")
async def api_image(filename: str):
    for d in Path().glob("Generated_Images_*"):
        candidate = d / filename
        if candidate.exists():
            return FileResponse(str(candidate), media_type="image/png")
    raise HTTPException(status_code=404, detail="Image not found")


@app.post("/api/open-folder")
async def api_open_folder(payload: dict):
    filename = (payload or {}).get("filename", "")
    if not filename:
        raise HTTPException(status_code=400, detail="filename required")
    import subprocess
    import platform
    for d in Path().glob("Generated_Images_*"):
        candidate = d / filename
        if candidate.exists():
            folder = str(d.resolve())
            try:
                if platform.system() == "Windows":
                    os.startfile(folder)
                elif platform.system() == "Darwin":
                    subprocess.run(["open", folder])
                else:
                    subprocess.run(["xdg-open", folder])
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))
            return {"status": "ok", "folder": folder}
    raise HTTPException(status_code=404, detail="Image not found")


# ---------------------------------------------------------------------------
#  Static file serving (SPA fallback)
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index():
    return FileResponse(str(static_dir / "index.html"), media_type="text/html")


if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    @app.get("/static/{path:path}")
    async def static_fallback(path: str):
        raise HTTPException(status_code=404, detail="Static files not found")


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main():
    global _cache_dir

    _setup_hardware()

    import torch
    if not torch.cuda.is_available():
        print("\n  WARNING: CUDA not available. Image generation will fail.")
        print("  Install PyTorch with CUDA from https://pytorch.org/get-started/locally/\n")

    _cache_dir = _resolve_cache_dir()
    if not _cache_dir:
        print("\n  WARNING: No cache directory found. Run 'python dapple.py --setup' first.\n")
    else:
        os.environ["HF_HOME"] = _cache_dir

    import argparse
    parser = argparse.ArgumentParser(description="Dapple Web Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print()
    print("  Dapple Web Server")
    print(f"  Local:   http://localhost:{args.port}")
    print(f"  Network: http://{args.host}:{args.port}")
    print(f"  Cache:   {_cache_dir or 'not configured'}")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
