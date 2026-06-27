# Dapple Web UI — Setup Guide

The web UI wraps the same FLUX pipeline as the terminal version. See `Setup.md` for model details, VRAM estimates, and `.env` configuration.

---

## 1. Requirements

Same as the terminal version plus:

```
fastapi >= 0.115.0
uvicorn[standard] >= 0.30.0
pydantic >= 2.0.0
```

Install everything:

```
pip install -r requirements.txt
```

---

## 2. Start the server

```
python server.py --port 8000
```

Open `http://localhost:8000` in a browser. The terminal version (`dapple.py`) and web UI share the same `generator_config.json` — settings like `cache_dir` and `use_nf4` are synchronized between both.

---

## 3. Interface

### Sidebar (left)

| Section | Purpose |
|---|---|
| **Model** | Click a model card to load it. Spinner appears while loading, green checkmark when ready. Other models grey out during load. |
| **Cache Directory** | Type a path and click **Save**. Requires server restart to take effect. Falls back to `dapple.py --setup` if empty. |
| **Memory** | Toggle **NF4 Quantization** on/off. Click the **(i)** icon for an explanation. Takes effect on the next model load. |
| **Status** | Shows GPU name, VRAM, loaded model, and cache directory. |

Drag the right edge of the sidebar to resize it.

### Main area

| Element | Purpose |
|---|---|
| **Output panel** | Displays generated images. Shows a placeholder until the first image, then a pixel-wave animation during generation. |
| **Progress bar** | Shows real-time step progress while generating. |
| **Prompt input** | Type your prompt. Press **Enter** to generate (Shift+Enter for newline). |
| **Advanced** | Toggle to reveal negative prompt, steps, guidance, seed, width, and height inputs. |

### Result header

Once an image is generated, the header shows the filename and two buttons:

| Button | Action |
|---|---|
| **Folder** | Opens the `Generated_Images_YYYY-MM-DD/` folder in your file manager. |
| **X-icon** | Clears the image from view. Does not delete the file. |

---

## 4. Generation flow

1. Click a model in the sidebar (e.g. `FLUX.2 Klein 4B`).
2. Wait for the green checkmark.
3. Type a prompt in the textarea at the bottom.
4. Enter optional settings via the **Advanced** toggle.
5. Press **Enter** or click the send button.
6. Watch the progress bar and pixel animation.
7. The image appears in the output panel.
8. Click the folder icon to open the output directory, or the X to clear.
9. Continue to generate as many images as you want, its free :D

---

## 5. Config synchronization

The web UI and terminal version share `generator_config.json`:

| Setting | Persisted to config? |
|---|---|
| `cache_dir` (Cache Directory input) | Yes |
| `use_nf4` (NF4 toggle) | Yes |
| Steps, guidance, width, height, seed | No — sent per-generation in the request body |
| `model`, `default_prompt`, `default_negative_prompt` | Read from config but not editable in web UI |

