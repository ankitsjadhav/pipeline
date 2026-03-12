import os
import sys
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix for Colab multiprocessing / "No module named '__main__'" issue
if "__main__" not in sys.modules:
    import types
    sys.modules["__main__"] = types.ModuleType("__main__")

# Lazy import: do NOT import torch or diffusers here. They are imported inside
# load_flux_model() so Colab's pre-installed torch is used once, when FLUX loads.

import gc
import time
from pathlib import Path

import numpy as np
from PIL import Image

from utils import ensure_dir


HARDCODED_SUFFIX = (
    ", cinematic lighting, photorealistic, 8k quality, "
    "no text, no letters, no words, no writing, no symbols, "
    "no signs, no watermark, no numbers, no charts, no calendar, "
    "no newspaper, no book pages, no scroll text, no banners, "
    "no posters, no zodiac wheels, no UI elements"
)
QUALITY_ADDITION = "professional photography, sharp focus, high detail"


def get_dimensions(output_format: str) -> tuple[int, int]:
    formats = {
        "reels": (1080, 1920),
        "square": (1080, 1080),
        "landscape": (1920, 1080),
    }
    return formats.get(output_format, (1080, 1920))


def load_flux_model(config):
    """
    Load FLUX.1 Schnell. Torch and diffusers are imported only here (single load).
    Colab: run Cell 1 once, restart runtime, then run only Cell 2. Do not import torch in the notebook.
    """
    # Ensure env before any torch/transformers/diffusers import (avoids Colab tokenizer/torch conflicts)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch
    except Exception as e:
        raise RuntimeError(
            "Failed to import torch. In Colab: run Cell 1, then Runtime → Restart runtime, then run only Cell 2. Do not pip install torch."
        ) from e
    try:
        from diffusers import FluxPipeline
    except Exception as e:
        raise RuntimeError(
            "Failed to import diffusers. Restart runtime after Cell 1, then run only Cell 2. If error persists, try: pip install -U diffusers transformers."
        ) from e

    print("Loading FLUX.1 Schnell (bfloat16, CPU offload)...", flush=True)
    print(f"  Torch {getattr(torch, '__version__', '?')} | CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}", flush=True)
    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except (AssertionError, RuntimeError) as e:
        err_msg = str(e).lower()
        if "docstring" in err_msg or "dlpack" in err_msg or "already" in err_msg:
            raise RuntimeError(
                "Colab torch/diffusers conflict. Run Cell 1 once, then Runtime → Restart runtime, then run only Cell 2. Do not import torch in the notebook."
            ) from e
        raise
    print("✅ FLUX loaded successfully", flush=True)
    return pipe


def _is_black_image(pil_image) -> bool:
    arr = np.array(pil_image)
    return float(arr.mean()) < 10.0


def generate_image(pipe, scene: dict, config: dict, images_dir: Path) -> Path | None:
    """
    Generate one FLUX image for Type B/C scenes (Type A returns None).
    Always saves to Drive immediately on success.
    """
    if scene.get("type") == "A":
        return None

    flux_prompt = scene.get("flux_prompt")
    if not flux_prompt or not isinstance(flux_prompt, str):
        raise RuntimeError(f"Scene {scene.get('id')} missing flux_prompt for type {scene.get('type')}")

    # CLIP 77-token limit: truncate to 200 chars to avoid silent cutoff
    flux_prompt = flux_prompt[:200]
    prompt = flux_prompt + HARDCODED_SUFFIX + ", " + QUALITY_ADDITION

    import torch
    scene_id = scene.get("id")
    if scene_id is None:
        scene_id = 0
    try:
        scene_id = int(scene_id)
    except (TypeError, ValueError):
        scene_id = 0
    images_dir_p = Path(str(images_dir))
    out_path = images_dir_p / f"scene_{scene_id:02d}.png"
    ensure_dir(Path(out_path.parent))

    # T4-safe: generate at small resolution only; never pass 1920x1080 to FLUX
    FALLBACK_RESOLUTIONS = [(768, 432), (640, 360), (512, 288)]
    UPSCALE_SIZE = (1920, 1080)

    last_err: Exception | None = None
    image = None
    for (gen_w, gen_h) in FALLBACK_RESOLUTIONS:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            result = pipe(
                prompt=prompt,
                width=gen_w,
                height=gen_h,
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=torch.Generator("cuda").manual_seed(int(time.time())),
            )
            image = result.images[0]
            if _is_black_image(image):
                print(f"Black image detected (scene {scene_id}) at {gen_w}x{gen_h}, trying next resolution...", flush=True)
                continue
            break
        except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
            print(f"OOM at {gen_w}x{gen_h} (scene {scene_id}), clearing cache and trying next size...", flush=True)
            last_err = e
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            last_err = e
            torch.cuda.empty_cache()
            gc.collect()
            continue

    if image is None:
        raise RuntimeError(
            f"FLUX generation failed for scene {scene_id} after all resolution fallbacks (768x432, 640x360, 512x288). Last error: {last_err}"
        )

    # Upscale to 1920x1080 with PIL (never pass 1920x1080 to the pipeline)
    image = image.resize(UPSCALE_SIZE, Image.Resampling.LANCZOS)

    try:
        image.save(str(out_path))
    except Exception as e1:
        tmp = Path("/content/images")
        ensure_dir(tmp)
        tmp_path = Path(str(tmp)) / out_path.name
        try:
            image.save(str(tmp_path))
        except Exception as e2:
            raise RuntimeError(f"Failed to save FLUX image to Drive or /content: {e1!r}; fallback: {e2!r}") from e2
        try:
            import shutil
            shutil.copy2(str(Path(tmp_path)), str(Path(out_path)))
        except Exception:
            Path(out_path).write_bytes(Path(tmp_path).read_bytes())
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
    print(f"Scene {scene_id} image saved: {out_path}", flush=True)

    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
    return out_path

