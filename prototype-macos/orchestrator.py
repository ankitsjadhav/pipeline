from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote_plus

import requests


ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT / "prompts"
OUTPUT_DIR = ROOT / "output"


def require_bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(
            f"Missing dependency: `{name}` not found on PATH.\n"
            f"Install it and try again."
        )
    return path


def read_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _try_start_ollama_server(wait_s: float = 6.0) -> None:
    """
    Best-effort attempt to start the local Ollama server.
    If it's already running, starting may fail harmlessly.
    """
    if not shutil.which("ollama"):
        return
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(wait_s)
    except Exception:
        return


def _ollama_healthcheck(timeout_s: float = 0.5) -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=timeout_s)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _ensure_ollama_running() -> None:
    if _ollama_healthcheck():
        return
    if not shutil.which("ollama"):
        raise RuntimeError("`ollama` CLI not found on PATH (install Ollama first)")
    _try_start_ollama_server()
    if _ollama_healthcheck():
        return
    raise RuntimeError("Ollama is not reachable at http://localhost:11434")


def _ollama_parse_generate_response(resp: requests.Response) -> str:
    if resp.status_code != 200:
        raise SystemExit(f"Ollama error ({resp.status_code}): {resp.text[:500]}")
    try:
        data = resp.json()
    except json.JSONDecodeError:
        raise SystemExit(f"Unexpected Ollama response: {resp.text[:500]}")
    out = (data.get("response") or "").strip()
    if not out:
        raise SystemExit("Ollama returned an empty response.")
    return out


def parse_loose_json(text: str) -> dict:
    """
    Best-effort JSON extraction for LLM outputs.
    Tries strict parse first, then extracts the outermost {...} block.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise SystemExit("Model did not return JSON. Try again or change the model/prompt.")

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise SystemExit(f"Could not parse JSON from model output: {e}")


def ollama_generate(*, model: str, prompt: str, timeout_s: int = 180) -> str:
    url = "http://localhost:11434/api/generate"
    try:
        _ensure_ollama_running()
    except RuntimeError as e:
        raise SystemExit(
            "Could not reach Ollama at `http://localhost:11434`.\n"
            "Make sure Ollama is running:\n"
            "  - `ollama serve`\n"
            "  - or open the Ollama app\n\n"
            f"Original error: {e}"
        )

    try:
        resp = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout_s,
        )
        return _ollama_parse_generate_response(resp)
    except requests.RequestException as e:
        # One retry after attempting to start Ollama.
        _try_start_ollama_server()
        try:
            resp = requests.post(
                url,
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=timeout_s,
            )
            return _ollama_parse_generate_response(resp)
        except requests.RequestException:
            raise SystemExit(
                "Could not reach Ollama at `http://localhost:11434`.\n"
                "Make sure Ollama is running:\n"
                "  - `ollama serve`\n"
                "  - or open the Ollama app\n\n"
                f"Original error: {e}"
            )


def pocket_tts_generate(*, text: str, output_wav: Path, voice: str) -> None:
    require_bin("pocket-tts")
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    # `pocket-tts generate` accepts text inline; keep it single arg.
    cmd = [
        "pocket-tts",
        "generate",
        "--text",
        text,
        "--voice",
        voice,
        "--output-path",
        str(output_wav),
        "--quiet",
    ]
    subprocess.run(cmd, check=True)


def say_tts_generate(*, text: str, output_wav: Path) -> None:
    """Fallback: use macOS built-in `say` to produce voiceover (no voice cloning)."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    txt_path = output_wav.with_suffix(".say.txt")
    aiff_path = output_wav.with_suffix(".aiff")
    try:
        txt_path.write_text(text, encoding="utf-8")
        subprocess.run(
            ["say", "-f", str(txt_path), "-o", str(aiff_path)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(aiff_path),
                "-acodec",
                "pcm_s16le",
                "-ar",
                "24000",
                str(output_wav),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        for p in (txt_path, aiff_path):
            if p.exists():
                p.unlink(missing_ok=True)


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def _ffmpeg_subtitles_filter(srt_path: Path) -> str:
    """Build subtitles filter with path escaped for FFmpeg (avoids parsing errors on macOS)."""
    path_str = str(srt_path.resolve())
    # Single-quote wrap so path is one token; escape any ' in path as '\''
    path_str = path_str.replace("'", "'\\''")
    return f"subtitles='{path_str}'"


def format_srt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def whisper_word_srt(*, audio_path: Path, srt_path: Path, model_name: str) -> None:
    from faster_whisper import WhisperModel

    srt_path.parent.mkdir(parents=True, exist_ok=True)

    # CPU-friendly defaults for macOS.
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(audio_path), word_timestamps=True)

    idx = 1
    with srt_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            if not getattr(seg, "words", None):
                continue
            for w in seg.words:
                word = (w.word or "").strip()
                if not word:
                    continue
                start = format_srt_ts(float(w.start))
                end = format_srt_ts(float(w.end))
                f.write(f"{idx}\n{start} --> {end}\n{word}\n\n")
                idx += 1


def ffmpeg_make_vertical_video(*, audio_wav: Path, srt_path: Path, out_mp4: Path) -> None:
    require_bin("ffmpeg")

    dur = wav_duration_seconds(audio_wav)
    # Give subtitles a little tail at the end.
    dur = max(1.0, dur + 0.25)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    subtitles_filter = _ffmpeg_subtitles_filter(srt_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=1080x1920:r=30",
        "-i",
        str(audio_wav),
        "-t",
        f"{dur:.3f}",
        "-vf",
        subtitles_filter,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


# --- B-roll: Pexels API or placeholder images → Ken Burns clips → final assembly ---

BROLL_SIZE = (1080, 1920)  # vertical
BROLL_FPS = 30
BROLL_CLIP_DURATION = 3  # seconds per clip
XFADE_DURATION = 0.35  # crossfade between clips (seconds)
XFADE_TRANSITIONS = ("fade", "wipeleft", "wiperight", "slideleft", "slideright", "wipeup", "wipedown", "fade")
MAX_BROLL_SHOTS = 6  # 6–8 scenes per ad to limit render load; keeps all animations
PEXELS_PER_QUERY = 5  # fetch multiple, rank by orientation/resolution
CACHE_DIR_NAME = ".ugc_cache"  # under output dir for assets
# Soft background colors for placeholder fallback (hex)
BROLL_COLORS = ("#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560", "#2d4059", "#3d5a80", "#ee6c4d")

PEXELS_API_URL = "https://api.pexels.com/v1/search"

# Stopwords to drop so the Pexels query keeps visual keywords only
_PEXELS_STOPWORDS = frozenset({
    "the", "a", "an", "of", "on", "with", "to", "into", "and", "or", "for", "in", "at", "by",
    "as", "from", "that", "this", "it", "is", "are", "was", "were", "been", "being", "their",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "shall", "can", "need", "used", "what", "which", "when", "where", "who", "how",
})


def _extract_product_keywords(product_name: str, product_description: str, max_words: int = 4) -> str:
    """
    Extract distinctive product keywords for Pexels (e.g. astrology, horoscope, birth chart).
    Ensures fetched images match the ad's product/theme.
    """
    combined = f"{product_name} {product_description}".lower()
    for c in ".,;:!?\"'()[]-–—":
        combined = combined.replace(c, " ")
    words = [w for w in combined.split() if w and len(w) > 2]
    # Drop generic terms; keep distinctive product/theme words
    generic = _PEXELS_STOPWORDS | {"app", "product", "daily", "using", "made", "get", "now"}
    kept = [w for w in words if w not in generic]
    # Dedupe while preserving order
    seen = set()
    unique = []
    for w in kept:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return " ".join(unique[:max_words]) if unique else ""


# Visual terms for Pexels (order = relevance for stock photos)
_VISUAL_TERMS = (
    "person", "woman", "man", "people", "portrait", "hand", "hands", "close up",
    "phone", "smartphone", "mobile", "screen", "app", "typing", "checking", "looking",
    "holding", "reading", "scrolling", "laptop", "lifestyle",
)


def _shot_to_pexels_query(
    shot_text: str,
    product_keywords: str = "",
    max_shot_words: int = 12,
) -> str:
    """
    Build Pexels query: product keywords + shot description + visual terms.
    Order improves relevance: product context first, then shot-specific, then visual.
    """
    product_part = (product_keywords or "").strip()
    if not shot_text or not shot_text.strip():
        shot_part = "person smartphone portrait"
    else:
        text = shot_text.lower().strip()
        for c in ".,;:!?\"'()[]-–—":
            text = text.replace(c, " ")
        words = [w for w in text.split() if w and w not in _PEXELS_STOPWORDS and len(w) > 1]
        ordered = [w for w in _VISUAL_TERMS if w in words]
        ordered += [w for w in words if w not in _VISUAL_TERMS]
        shot_part = " ".join(ordered[:max_shot_words]) if ordered else "person smartphone"
    combined = f"{product_part} {shot_part}".strip() if product_part else shot_part
    return combined or "person smartphone portrait"


def _fetch_pexels_photos(
    shot_text: str,
    api_key: str,
    product_keywords: str,
    page: int,
    per_page: int = PEXELS_PER_QUERY,
) -> list[dict]:
    """Fetch multiple Pexels results for a query. Returns list of photo dicts (src, width, height)."""
    if not api_key:
        return []
    query = _shot_to_pexels_query(shot_text, product_keywords)
    q = quote_plus(query[:200])
    url = f"{PEXELS_API_URL}?query={q}&per_page={per_page}&orientation=portrait&page={max(1, page)}"
    try:
        r = requests.get(url, headers={"Authorization": api_key}, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("photos") or []
    except (requests.RequestException, KeyError):
        return []


def _rank_pexels_photo(photo: dict) -> tuple[int, int]:
    """Score for ranking: higher is better. (portrait fit, resolution)."""
    src = photo.get("src") or {}
    w = photo.get("width") or src.get("width") or 0
    h = photo.get("height") or src.get("height") or 0
    # Prefer portrait or near-portrait (height >= width)
    orientation_ok = 1 if (h >= w * 0.9) else 0
    # Prefer larger resolution for quality
    pixels = w * h if w and h else 0
    return (orientation_ok, pixels)


def _fetch_pexels_best_photo(
    shot_text: str,
    api_key: str,
    product_keywords: str,
    page: int,
) -> bytes | None:
    """Fetch PEXELS_PER_QUERY results, rank by orientation and resolution, return best image bytes."""
    photos = _fetch_pexels_photos(shot_text, api_key, product_keywords, page)
    if not photos:
        return None
    # Rank and pick best
    ranked = [(p, _rank_pexels_photo(p)) for p in photos]
    ranked.sort(key=lambda x: x[1], reverse=True)
    best = ranked[0][0]
    img_url = (
        best.get("src", {}).get("large")
        or best.get("src", {}).get("original")
    )
    if not img_url:
        return None
    try:
        r = requests.get(img_url, timeout=20)
        r.raise_for_status()
        return r.content
    except requests.RequestException:
        return None


def _resize_and_crop_bytes_to_broll(image_bytes: bytes, output_path: Path) -> bool:
    """Load image from bytes, center-crop to 1080x1920, save as PNG. Returns True on success."""
    try:
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        target_w, target_h = BROLL_SIZE
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "PNG")
        return True
    except Exception:
        return False


def _cache_key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _cache_path(cache_dir: Path, subdir: str, key: str, ext: str) -> Path:
    p = cache_dir / subdir / f"{key}{ext}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _get_shotlist(assets: dict) -> list[dict]:
    """Extract broll_shotlist from assets; each item has 'time' and 'shot'."""
    shotlist = assets.get("broll_shotlist") or []
    out = []
    for item in shotlist:
        if not isinstance(item, dict):
            continue
        shot = str(item.get("shot") or "").strip()
        if not shot:
            continue
        out.append({"time": str(item.get("time") or "").strip(), "shot": shot})
    return out


def _generate_placeholder_image(shot_text: str, output_path: Path, color_hex: str) -> None:
    """Generate one 1080x1920 placeholder image with scene description text."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", BROLL_SIZE, color_hex)
    draw = ImageDraw.Draw(img)

    # Prefer system font on macOS; fallback to default
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    font_size = 48
    font = None
    for fp in font_paths:
        if Path(fp).exists():
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    # Wrap text to ~35 chars per line
    words = shot_text.split()
    lines = []
    current = []
    for w in words:
        current.append(w)
        if len(" ".join(current)) > 38:
            lines.append(" ".join(current))
            current = []
    if current:
        lines.append(" ".join(current))
    text_block = "\n".join(lines[:6])  # max 6 lines

    # Center text (bbox for multiline)
    try:
        bbox = draw.textbbox((0, 0), text_block, font=font)
    except AttributeError:
        try:
            bbox = draw.getbbox((0, 0), text_block, font=font)
        except (AttributeError, TypeError):
            bbox = (0, 0, 400, 120)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (BROLL_SIZE[0] - tw) // 2
    y = (BROLL_SIZE[1] - th) // 2
    draw.text((x, y), text_block, fill="white", font=font)
    img.save(output_path, "PNG")


def _fetch_and_preprocess_one(
    i: int,
    item: dict,
    product_keywords: str,
    api_key: str,
    broll_dir: Path,
    cache_dir: Path | None,
    debug: bool,
) -> Path:
    """Fetch (or load from cache) one image, preprocess to BROLL_SIZE, return path."""
    out_path = broll_dir / f"scene_{i:03d}.png"
    shot_text = item["shot"]
    cache_key = _cache_key(product_keywords, shot_text, str(i))
    if cache_dir:
        cached = _cache_path(cache_dir, "images", cache_key, ".png")
        if cached.exists():
            shutil.copy2(cached, out_path)
            return out_path
    if api_key and shot_text:
        raw = _fetch_pexels_best_photo(shot_text, api_key, product_keywords, page=i + 1)
        if raw and _resize_and_crop_bytes_to_broll(raw, out_path):
            if cache_dir:
                shutil.copy2(out_path, _cache_path(cache_dir, "images", cache_key, ".png"))
            return out_path
    color = BROLL_COLORS[i % len(BROLL_COLORS)]
    _generate_placeholder_image(shot_text, out_path, color)
    if cache_dir:
        shutil.copy2(out_path, _cache_path(cache_dir, "images", cache_key, ".png"))
    return out_path


def generate_broll_images(
    assets: dict,
    broll_dir: Path,
    product_name: str = "",
    product_description: str = "",
    cache_dir: Path | None = None,
    debug: bool = False,
) -> list[Path]:
    """
    Fetch (or use cache) one image per shot; rank Pexels results; preprocess to BROLL_SIZE.
    Runs download and preprocessing in parallel. 6–8 scenes (MAX_BROLL_SHOTS).
    """
    shotlist = _get_shotlist(assets)
    if not shotlist:
        shotlist = [{"time": "0-0s", "shot": "Ad"}]
    shotlist = shotlist[:MAX_BROLL_SHOTS]
    broll_dir.mkdir(parents=True, exist_ok=True)
    api_key = (os.environ.get("PEXELS_API_KEY") or "").strip()
    product_keywords = _extract_product_keywords(product_name, product_description)
    if debug:
        print(f"[DEBUG] B-roll: {len(shotlist)} shots, product_keywords='{product_keywords}'", flush=True)
    max_workers = min(len(shotlist), 6)
    paths = [None] * len(shotlist)  # type: list[Path | None]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                _fetch_and_preprocess_one,
                i, item, product_keywords, api_key, broll_dir, cache_dir, debug,
            ): i
            for i, item in enumerate(shotlist)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            paths[i] = fut.result()
    return [p for p in paths if p is not None]


# Predefined animation presets (zoom_in, zoom_out, pan_left, pan_right) for performance + variety
ANIMATION_PRESETS = (
    ("zoom_in", "min(zoom+0.0006,1.12)", "iw/2-(iw/zoom/2)+25*sin(on/30)", "ih/2-(ih/zoom/2)"),
    ("zoom_out", "max(1.15-0.0005*on,1)", "iw/2-(iw/zoom/2)-20*sin(on/28)", "ih/2-(ih/zoom/2)"),
    ("pan_left", "min(zoom+0.0005,1.1)", "iw/2-(iw/zoom/2)-30*sin(on/32)", "ih/2-(ih/zoom/2)"),
    ("pan_right", "min(zoom+0.0006,1.1)", "iw/2-(iw/zoom/2)+25*sin(on/30)", "ih/2-(ih/zoom/2)"),
)
_ZOOM_PRESETS = [p[1:] for p in ANIMATION_PRESETS]  # (z, x, y) only for backward compat


def compute_scene_timeline(n_scenes: int) -> list[tuple[float, float]]:
    """Pre-calculate (start_sec, end_sec) for each scene so FFmpeg uses one pass."""
    if n_scenes <= 0:
        return []
    step = BROLL_CLIP_DURATION - XFADE_DURATION
    return [(i * step, i * step + BROLL_CLIP_DURATION) for i in range(n_scenes)]


def _detect_gpu_encoder() -> str:
    """Use h264_nvenc when available (cloud/GPU), else libx264."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and "h264_nvenc" in (out.stdout or ""):
            return "h264_nvenc"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "libx264"


def _image_to_animated_clip(
    image_path: Path,
    output_mp4: Path,
    duration_sec: float = BROLL_CLIP_DURATION,
    clip_index: int = 0,
) -> None:
    """Render one image as a vertical clip with Ken Burns (zoom + pan). Varies style per clip_index."""
    require_bin("ffmpeg")
    frames = max(1, int(round(duration_sec * BROLL_FPS)))
    preset = _ZOOM_PRESETS[clip_index % len(_ZOOM_PRESETS)]
    z_expr, x_expr, y_expr = preset
    zoompan = (
        f"zoompan=z='{z_expr}':"
        f"x='{x_expr}':"
        f"y='{y_expr}':"
        f"d={frames}:s={BROLL_SIZE[0]}x{BROLL_SIZE[1]}:fps={BROLL_FPS}"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-vf",
        zoompan,
        "-t",
        f"{duration_sec:.3f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(BROLL_FPS),
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def broll_images_to_clips(image_paths: list[Path], broll_dir: Path) -> list[Path]:
    """Convert each B-roll image to an animated clip (Ken Burns) of BROLL_CLIP_DURATION seconds."""
    if not image_paths:
        return []
    clip_paths = []
    for i, img_path in enumerate(image_paths):
        out_mp4 = broll_dir / f"clip_{i:03d}.mp4"
        _image_to_animated_clip(img_path, out_mp4, clip_index=i)
        clip_paths.append(out_mp4)
    return clip_paths


def assemble_final_with_broll(
    clip_paths: list[Path],
    audio_wav: Path,
    srt_path: Path,
    out_mp4: Path,
) -> None:
    """Concat B-roll clips with crossfade transitions, add voiceover, burn captions."""
    require_bin("ffmpeg")
    if not clip_paths:
        ffmpeg_make_vertical_video(audio_wav=audio_wav, srt_path=srt_path, out_mp4=out_mp4)
        return
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    n = len(clip_paths)
    dur = wav_duration_seconds(audio_wav)
    dur = max(1.0, dur + 0.25)

    # Build xfade chain: [0:v][1:v]xfade...[v1]; [v1][2:v]xfade...[v2]; ...
    fd = XFADE_DURATION
    clip_dur = BROLL_CLIP_DURATION
    offset_step = clip_dur - fd
    filter_parts = []
    for i in range(n - 1):
        left = f"[v{i}]" if i > 0 else "[0:v]"
        right = f"[{i+1}:v]"
        offset = offset_step * (i + 1)
        out = f"[v{i+1}]"
        transition = XFADE_TRANSITIONS[i % len(XFADE_TRANSITIONS)]
        filter_parts.append(f"{left}{right}xfade=transition={transition}:duration={fd}:offset={offset:.2f}{out}")
    filter_chain = ";".join(filter_parts)
    last_v = f"[v{n-1}]" if n > 1 else "[0:v]"
    subtitles_filter = _ffmpeg_subtitles_filter(srt_path)
    filter_chain += f";{last_v}{subtitles_filter}[vout]"

    cmd = ["ffmpeg", "-y"]
    for c in clip_paths:
        cmd.extend(["-i", str(c)])
    cmd.extend(["-i", str(audio_wav)])
    cmd.extend([
        "-filter_complex", filter_chain,
        "-map", "[vout]",
        "-map", f"{n}:a:0",
        "-t", f"{dur:.3f}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(out_mp4),
    ])
    subprocess.run(cmd, check=True)


def assemble_single_pass_with_broll(
    image_paths: list[Path],
    audio_wav: Path,
    srt_path: Path,
    out_mp4: Path,
) -> None:
    """
    Single FFmpeg render pass: all images → zoompan (predefined presets) → xfade → subtitles.
    Pre-calculated timeline; GPU encoding (h264_nvenc) when available.
    Images must already be preprocessed to BROLL_SIZE.
    """
    require_bin("ffmpeg")
    if not image_paths:
        ffmpeg_make_vertical_video(audio_wav=audio_wav, srt_path=srt_path, out_mp4=out_mp4)
        return
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    n = len(image_paths)
    dur = wav_duration_seconds(audio_wav)
    dur = max(1.0, dur + 0.25)
    fd = XFADE_DURATION
    clip_dur = BROLL_CLIP_DURATION
    offset_step = clip_dur - fd
    frames = max(1, int(round(clip_dur * BROLL_FPS)))
    encoder = _detect_gpu_encoder()

    # One filter_complex: each input -> scale (if needed) -> zoompan -> [v0],[v1],...
    # Then [v0][v1]xfade...[v1]; [v1][v2]xfade...[v2]; ... ; [v(n-1)]subtitles[vout]
    zoompan_parts = []
    for i in range(n):
        preset = _ZOOM_PRESETS[i % len(_ZOOM_PRESETS)]
        z_expr, x_expr, y_expr = preset
        zp = f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s={BROLL_SIZE[0]}x{BROLL_SIZE[1]}:fps={BROLL_FPS}"
        zoompan_parts.append(f"[{i}:v]{zp}[v{i}]")
    filter_chain = ";".join(zoompan_parts)

    for i in range(n - 1):
        left = f"[v{i}]" if i > 0 else "[v0]"
        right = f"[v{i+1}]"
        offset = offset_step * (i + 1)
        trans = XFADE_TRANSITIONS[i % len(XFADE_TRANSITIONS)]
        filter_chain += f";{left}{right}xfade=transition={trans}:duration={fd}:offset={offset:.2f}[v{i+1}]"
    last_v = f"[v{n-1}]"
    subs = _ffmpeg_subtitles_filter(srt_path)
    filter_chain += f";{last_v}{subs}[vout]"

    cmd = ["ffmpeg", "-y"]
    for p in image_paths:
        cmd.extend(["-loop", "1", "-i", str(p)])
    cmd.extend(["-i", str(audio_wav)])
    cmd.extend([
        "-filter_complex", filter_chain,
        "-map", "[vout]", "-map", f"{n}:a:0",
        "-t", f"{dur:.3f}",
        "-c:v", encoder,
        "-pix_fmt", "yuv420p",
    ])
    if encoder == "h264_nvenc":
        cmd.extend(["-preset", "p4", "-b:v", "5M"])
    cmd.extend(["-c:a", "aac", "-shortest", str(out_mp4)])
    subprocess.run(cmd, check=True)


def assemble_via_cloud(
    image_paths: list[Path],
    audio_wav: Path,
    srt_path: Path,
    out_mp4: Path,
    worker_url: str,
) -> None:
    """Upload images + voiceover + captions to cloud worker; download final.mp4."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    files: list[tuple[str, tuple[str, bytes]]] = []
    for p in sorted(image_paths):
        files.append(("images", (p.name, p.read_bytes())))
    files.append(("voiceover", ("voiceover.wav", audio_wav.read_bytes())))
    files.append(("captions", ("captions.srt", srt_path.read_bytes())))
    r = requests.post(worker_url, files=files, timeout=300)
    r.raise_for_status()
    out_mp4.write_bytes(r.content)


def _render_numbered_section(title: str, items: list) -> str:
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    if not cleaned:
        return ""
    lines = [f"{title}\n"]
    lines.extend(f"{i}. {t}\n" for i, t in enumerate(cleaned, 1))
    return "".join(lines)


def _render_shotlist_section(title: str, shotlist: list) -> str:
    if not shotlist:
        return ""
    lines = [f"{title}\n"]
    for item in shotlist:
        if not isinstance(item, dict):
            continue
        t = str(item.get("time") or "").strip()
        s = str(item.get("shot") or "").strip()
        if t or s:
            lines.append(f"- **{t or 'time'}**: {s}\n")
    return "".join(lines) if len(lines) > 1 else ""


def write_edit_plan_md(*, out_path: Path, product: str, assets: dict) -> None:
    hooks = assets.get("hooks") or []
    overlays = assets.get("text_overlays") or []
    shotlist = assets.get("broll_shotlist") or []
    cta = str(assets.get("cta") or "").strip()

    body = []
    body.append("# Edit plan (CapCut-friendly)\n")
    body.append(f"**Product**: {product}\n\n")
    body.append(_render_numbered_section("## Hooks (pick 1)", hooks))
    body.append("\n" if body[-1] else "")
    body.append(_render_numbered_section("## Text overlays (sprinkle through)", overlays))
    body.append("\n" if body[-1] else "")
    body.append(_render_shotlist_section("## B-roll shotlist", shotlist))
    if cta:
        body.append("\n## CTA\n")
        body.append(f"{cta}\n")

    out_path.write_text("".join(body).rstrip() + "\n", encoding="utf-8")

def build_brief(args: argparse.Namespace) -> dict:
    return {
        "product": args.product.strip(),
        "description": args.description.strip(),
        "audience": args.audience.strip(),
        "ollama_model": args.ollama_model,
        "tts_voice": args.tts_voice,
        "whisper_model": args.whisper_model,
        "assets_prompt": args.assets_prompt,
    }

def generate_ad_assets(*, model: str, prompt_path: Path, brief: dict) -> dict:
    prompt_tmpl = read_template(prompt_path)
    prompt = prompt_tmpl.format(
        product_name=brief["product"],
        product_description=brief["description"],
        target_audience=brief["audience"],
    )
    raw = ollama_generate(model=model, prompt=prompt)
    return parse_loose_json(raw)

def save_ad_assets(*, out_dir: Path, product: str, assets: dict) -> Path:
    assets_path = out_dir / "ad_assets.json"
    assets_path.write_text(json.dumps(assets, indent=2) + "\n", encoding="utf-8")

    hooks = assets.get("hooks") or []
    if isinstance(hooks, list) and hooks:
        (out_dir / "hooks.txt").write_text(
            "\n".join(str(h).strip() for h in hooks if str(h).strip()) + "\n",
            encoding="utf-8",
        )

    edit_plan_path = out_dir / "edit_plan.md"
    write_edit_plan_md(out_path=edit_plan_path, product=product, assets=assets)
    return edit_plan_path

def get_voiceover_text(assets: dict) -> str:
    voiceover = str(assets.get("voiceover") or "").strip()
    if not voiceover:
        raise SystemExit("Model JSON missing required key: `voiceover`.")
    return voiceover

def run_av_pipeline(
    *,
    out_dir: Path,
    voiceover: str,
    tts_voice: str,
    whisper_model: str,
    assets: dict,
    product_name: str = "",
    product_description: str = "",
    debug: bool = False,
    use_cloud: bool = False,
    cloud_worker_url: str = "",
) -> tuple[Path, Path, Path]:
    voiceover_path = out_dir / "voiceover.txt"
    voiceover_path.write_text(voiceover + "\n", encoding="utf-8")

    cache_dir = out_dir / CACHE_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / "voiceover.wav"
    audio_cache_key = _cache_key(voiceover, tts_voice)
    cached_audio = _cache_path(cache_dir, "audio", audio_cache_key, ".wav") if cache_dir else None
    if cached_audio and cached_audio.exists():
        shutil.copy2(cached_audio, audio_path)
    else:
        if shutil.which("pocket-tts"):
            print("Generating voiceover audio (Pocket TTS)...")
            pocket_tts_generate(text=voiceover, output_wav=audio_path, voice=tts_voice)
        elif shutil.which("say") and shutil.which("ffmpeg"):
            print("Generating voiceover audio (macOS `say` fallback)...")
            say_tts_generate(text=voiceover, output_wav=audio_path)
        else:
            raise SystemExit(
                "Missing TTS: install Pocket TTS or ensure `say` and `ffmpeg` are on PATH."
            )
        if cache_dir:
            shutil.copy2(audio_path, cached_audio)

    captions_path = out_dir / "captions.srt"
    captions_cache_key = _cache_key(voiceover, whisper_model)
    cached_captions = _cache_path(cache_dir, "captions", captions_cache_key, ".srt") if cache_dir else None
    if cached_captions and cached_captions.exists():
        shutil.copy2(cached_captions, captions_path)
    else:
        print("Generating captions (faster-whisper)...")
        whisper_word_srt(audio_path=audio_path, srt_path=captions_path, model_name=whisper_model)
        if cache_dir:
            shutil.copy2(captions_path, cached_captions)

    broll_dir = out_dir / "broll"
    broll_dir.mkdir(parents=True, exist_ok=True)
    print("Fetching B-roll images (Pexels, parallel + cache)...")
    image_paths = generate_broll_images(
        assets, broll_dir,
        product_name=product_name,
        product_description=product_description,
        cache_dir=cache_dir,
        debug=debug,
    )
    final_path = out_dir / "final.mp4"
    if use_cloud and cloud_worker_url:
        print("Uploading to cloud worker (B-roll clip rendering + assembly)...")
        assemble_via_cloud(
            image_paths=image_paths,
            audio_wav=audio_path,
            srt_path=captions_path,
            out_mp4=final_path,
            worker_url=cloud_worker_url,
        )
    else:
        print("Assembling final video (single pass, Ken Burns + xfade)...")
        assemble_single_pass_with_broll(
            image_paths=image_paths,
            audio_wav=audio_path,
            srt_path=captions_path,
            out_mp4=final_path,
        )
    return voiceover_path, audio_path, final_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal local UGC ad prototype (macOS).")
    parser.add_argument("--product", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--audience", required=True)
    parser.add_argument("--ollama-model", default="qwen2.5:7b-instruct-q4_K_M")
    parser.add_argument(
        "--tts-voice",
        default="hf://kyutai/tts-voices/alba-mackenna/casual.wav",
        help="Pocket TTS voice (HF URL or local audio path).",
    )
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument(
        "--assets-prompt",
        default="ad_factory_json.txt",
        help="Prompt filename in prototype-macos/prompts/ used to generate ad assets JSON.",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose logs for B-roll (Pexels fetch) stage.")
    parser.add_argument("--cloud", action="store_true", help="Use cloud worker for rendering.")
    parser.add_argument(
        "--cloud-worker-url",
        default=os.environ.get("UGC_CLOUD_WORKER_URL", ""),
        help="Cloud worker URL (default: UGC_CLOUD_WORKER_URL).",
    )
    args = parser.parse_args()

    if getattr(args, "cloud", False) and not (getattr(args, "cloud_worker_url", "") or "").strip():
        raise SystemExit("--cloud requires --cloud-worker-url or UGC_CLOUD_WORKER_URL.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    brief = build_brief(args)
    (OUTPUT_DIR / "brief.json").write_text(json.dumps(brief, indent=2) + "\n", encoding="utf-8")

    print("Generating script + edit assets with local LLM...")
    assets = generate_ad_assets(
        model=args.ollama_model,
        prompt_path=(PROMPTS_DIR / args.assets_prompt),
        brief=brief,
    )
    edit_plan_path = save_ad_assets(out_dir=OUTPUT_DIR, product=brief["product"], assets=assets)

    voiceover = get_voiceover_text(assets)
    voiceover_path, audio_path, final_path = run_av_pipeline(
        out_dir=OUTPUT_DIR,
        voiceover=voiceover,
        tts_voice=args.tts_voice,
        whisper_model=args.whisper_model,
        assets=assets,
        product_name=brief["product"],
        product_description=brief["description"],
        debug=getattr(args, "debug", False),
        use_cloud=getattr(args, "cloud", False),
        cloud_worker_url=(getattr(args, "cloud_worker_url", "") or "").strip(),
    )

    print("\nDone.")
    print(f"- {voiceover_path}")
    print(f"- {audio_path}")
    print(f"- {final_path}")
    print(f"- {edit_plan_path}")
    return 0


if __name__ == "__main__":
    # Unbuffer stdout/stderr in debug so logs appear during long SD runs
    if "--debug" in sys.argv:
        try:
            sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
            sys.stderr.reconfigure(line_buffering=True)  # type: ignore[union-attr]
        except AttributeError:
            pass
    raise SystemExit(main())

