import os
import sys
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from diffusers import FluxPipeline

import time
from pathlib import Path

import numpy as np

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
    print("Loading FLUX.1 Schnell in bfloat16 with CPU offload...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    print("✅ FLUX loaded successfully")
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

    prompt = flux_prompt + HARDCODED_SUFFIX + ", " + QUALITY_ADDITION

    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"torch missing; install dependencies first: {e}") from e

    width, height = get_dimensions(str(config.get("output_format") or "reels"))
    if not torch.cuda.is_available():
        print("WARNING: GPU not available. Reducing resolution for CPU runtime.", flush=True)
        width, height = (768, 768)

    out_path = images_dir / f"scene_{int(scene['id']):02d}.png"
    ensure_dir(out_path.parent)

    last_err: Exception | None = None
    for attempt in range(2):
        try:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=torch.Generator("cuda").manual_seed(int(time.time())),
            )
            image = result.images[0]
            if _is_black_image(image):
                print(f"Black image detected (scene {scene['id']}) attempt {attempt+1}, retrying...", flush=True)
                continue
            try:
                image.save(out_path)
            except Exception:
                # fallback to /content/images then copy
                tmp = Path("/content/images")
                ensure_dir(tmp)
                tmp_path = tmp / out_path.name
                image.save(tmp_path)
                tmp_path.replace(out_path)
            print(f"Scene {scene['id']} image saved: {out_path}", flush=True)
            return out_path
        except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
            print(f"OOM on attempt {attempt+1} (scene {scene['id']}), clearing cache and retrying...", flush=True)
            last_err = e
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pass
            time.sleep(3)
        except Exception as e:
            last_err = e
            time.sleep(2)

    raise RuntimeError(f"FLUX generation failed for scene {scene.get('id')} after 2 attempts: {last_err}")

