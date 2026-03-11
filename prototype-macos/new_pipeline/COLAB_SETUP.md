# Colab setup — verified Cell 1 & Cell 2

## Why the two-cell flow

1. **Cell 1** installs system and Python deps and mounts Drive. You then **restart the runtime** so torch/diffusers are not in a half-loaded state.
2. **Cell 2** runs only after restart: sets env, clones repo, imports pipeline, runs `run_pipeline()`. Torch and diffusers load only when `load_flux_model()` runs, avoiding `AssertionError` and "already has a docstring" errors.

**Do not** delete torch from `sys.modules`. **Do not** run `pip install torch` — use Colab’s pre-installed torch.

---

## Cell 1 — Install & mount (then restart runtime)

Run this cell, then **Runtime → Restart runtime**.

```python
import sys
import types
if "__main__" not in sys.modules:
    sys.modules["__main__"] = types.ModuleType("__main__")

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import subprocess
subprocess.run(["apt-get", "install", "-y", "ffmpeg", "libgl1", "fonts-noto"], capture_output=True)

# Do NOT install torch — use Colab's pre-installed torch (2.10+)
!pip install -q -U \
  diffusers \
  transformers \
  accelerate \
  groq \
  gTTS \
  faster-whisper \
  opencv-python-headless \
  scipy \
  sentencepiece \
  protobuf \
  huggingface_hub \
  ffmpeg-python \
  pydantic \
  langdetect \
  Pillow \
  numpy \
  requests

print("✅ Dependencies installed — now run Runtime → Restart runtime")
```

---

## Cell 2 — Run pipeline (after restart)

Run this cell only **after** you have run Cell 1 and restarted the runtime.

```python
import os
import sys
import shutil
import glob
import subprocess

# Env must be set before any pipeline import
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/content/drive/MyDrive/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/content/drive/MyDrive/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set HF_TOKEN for gated FLUX model (use Colab secrets or set here; do not commit)
# os.environ["HF_TOKEN"] = "your_hf_token"

if not os.path.exists("/content/pipeline"):
    subprocess.run(["git", "clone", "https://github.com/ankitsjadhav/pipeline.git", "/content/pipeline"])
else:
    subprocess.run(["git", "-C", "/content/pipeline", "pull"])

for root, dirs, files in os.walk("/content/pipeline"):
    for d in dirs:
        if d == "__pycache__":
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)

# Clear only pipeline modules (do NOT clear torch)
for mod in list(sys.modules.keys()):
    if any(x in mod for x in ["groq_director", "flux_generator", "smart_compositor", "ffmpeg_assembler", "main", "utils", "schemas", "setup_assets"]):
        del sys.modules[mod]

pipeline_path = "/content/pipeline/prototype-macos/new_pipeline"
if pipeline_path not in sys.path:
    sys.path.insert(0, pipeline_path)

print(f"✅ Pipeline ready: {os.listdir(pipeline_path)}")

import torch
print(f"✅ Torch {torch.__version__}")
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU — STOP HERE'}")

from main import CONFIG, run_pipeline

# Optional: override CONFIG before running (e.g. groq_api_key, drive_base_path)
# CONFIG["groq_api_key"] = "your_groq_key"
# CONFIG["drive_base_path"] = "/content/drive/MyDrive/ugc_pipeline"

output_dir = CONFIG.get("drive_base_path", "/content/drive/MyDrive/ugc_pipeline")
print(f"✅ Videos will be saved to: {output_dir}/output/")

run_pipeline()

# List generated videos (pipeline writes to {drive_base_path}/output/video_01.mp4, ...)
videos = glob.glob(os.path.join(output_dir, "output", "video_*.mp4"))
print(f"\n✅ DONE — {len(videos)} videos generated:")
for v in sorted(videos):
    print(f"  {v}")
```

---

## Pipeline code verification

Checked against the repo:

| Item | Status |
|------|--------|
| **Output path** | Videos are written to `{drive_base_path}/output/video_01.mp4` … `video_08.mp4`. Cell 2 glob `output_dir/output/video_*.mp4` is correct. |
| **CONFIG** | `main.py` uses `CONFIG["drive_base_path"]` and `CONFIG["groq_api_key"]`. Must be set (or overridden in Cell 2) before `run_pipeline()`. |
| **Folder structure** | `utils.create_folder_structure()` creates `base/output`, `base/images`, `base/composited`, `base/assets/mockups`, `base/assets/screenshots`, etc. Required assets: PNGs in `assets/mockups/` and `assets/screenshots/`. |
| **flux_generator** | Torch/diffusers are imported only inside `load_flux_model()` and `generate_image()` (lazy). No top-level torch import; safe with Colab’s pre-installed torch. |
| **main.py flow** | Mount → create folders → check space → check assets → load FLUX → for each video: plan (Groq) → generate images → composite → audio → captions → assemble. Skip if `output/video_XX.mp4` already exists. |
| **Versions** | No pinned versions in the pipeline. Colab’s torch (e.g. 2.10) + `pip install -U diffusers transformers accelerate ...` is fine. If you need to pin, use versions compatible with torch 2.10 (e.g. diffusers>=0.28.0, transformers>=4.40.0). |

---

## Optional: pin versions (if you see install conflicts)

If you get version conflicts, you can pin in Cell 1:

```bash
!pip install -q -U \
  diffusers>=0.28.0 \
  transformers>=4.40.0 \
  accelerate>=0.30.0 \
  groq gTTS faster-whisper opencv-python-headless scipy sentencepiece protobuf \
  huggingface_hub ffmpeg-python pydantic langdetect Pillow numpy requests
```

Still do **not** install or upgrade `torch`; use Colab’s pre-installed torch.
