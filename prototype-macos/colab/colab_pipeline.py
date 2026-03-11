"""
UGC Ad Factory — Google Colab pipeline (main).

Runs entirely in Colab: script (Ollama + Qwen 2.5 7B, same as local), voice (gTTS),
captions (faster-whisper), Pexels images, Ken Burns clips, FFmpeg assembly.
No external API keys for LLM — uses local Ollama in Colab. Heavy steps (image
processing, clip generation, animations, FFmpeg) run in Colab for better performance.

Usage: set PRODUCT, DESCRIPTION, AUDIENCE; set PEXELS_API_KEY and/or UNSPLASH_ACCESS_KEY for B-roll; call run_full_pipeline().
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import shutil
import subprocess
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Queue
from pathlib import Path
from urllib.parse import quote_plus

import requests

# --- Config (override via env or before import) ---
WORKDIR = Path(os.environ.get("UGC_COLAB_WORKDIR", "/content/ugc_ad_output"))
BROLL_SIZE = (1080, 1920)
BROLL_FPS = 30
BROLL_CLIP_DURATION = 3
XFADE_DURATION = 0.35
# Colab can handle longer timelines; do not cap by default (0 = unlimited).
MAX_BROLL_SHOTS = int(os.environ.get("UGC_MAX_BROLL_SHOTS", "0"))
PEXELS_PER_QUERY = 5
PEXELS_VIDEOS_URL = "https://api.pexels.com/videos/search"
WHISPER_MODEL = "base"  # smaller = less RAM, faster
PEXELS_API_URL = "https://api.pexels.com/v1/search"
UNSPLASH_API_URL = "https://api.unsplash.com/search/photos"
CACHE_DIR_NAME = ".ugc_cache"
ASSETS_JSON_FILENAME = "ad_assets.json"
VOICEOVER_TXT_FILENAME = "voiceover.txt"
VOICEOVER_WAV_FILENAME = "voiceover.wav"
CAPTIONS_SRT_FILENAME = "captions.srt"
FINAL_MP4_FILENAME = "final.mp4"

# Optional Stable Diffusion B-roll generation (runs in Colab when diffusers/torch are available)
SD_MODEL_ID = os.environ.get("UGC_SD_MODEL", "runwayml/stable-diffusion-v1-5")
SD_TIMEOUT_S = float(os.environ.get("UGC_SD_TIMEOUT_S", "120"))
SD_ENABLED = os.environ.get("UGC_SD_ENABLED", "1").strip() not in ("0", "false", "False")
SD_BATCH_SIZE = int(os.environ.get("UGC_SD_BATCH_SIZE", "2"))
# B-roll cache: "0" = always regenerate and overwrite; "1" = use cache when present (set UGC_BROLL_USE_CACHE=1 to enable)
BROLL_USE_CACHE = os.environ.get("UGC_BROLL_USE_CACHE", "0").strip().lower() in ("1", "true", "yes")

XFADE_TRANSITIONS = ("fade", "wipeleft", "wiperight", "slideleft", "slideright", "wipeup", "wipedown", "fade")
# Predefined animation presets (zoom_in, zoom_out, pan_left, pan_right) — same as local pipeline
ANIMATION_PRESETS = (
    ("zoom_in", "min(zoom+0.0006,1.12)", "iw/2-(iw/zoom/2)+25*sin(on/30)", "ih/2-(ih/zoom/2)"),
    ("zoom_out", "max(1.15-0.0005*on,1)", "iw/2-(iw/zoom/2)-20*sin(on/28)", "ih/2-(ih/zoom/2)"),
    ("pan_left", "min(zoom+0.0005,1.1)", "iw/2-(iw/zoom/2)-30*sin(on/32)", "ih/2-(ih/zoom/2)"),
    ("pan_right", "min(zoom+0.0006,1.1)", "iw/2-(iw/zoom/2)+25*sin(on/30)", "ih/2-(ih/zoom/2)"),
)
ZOOM_PRESETS = [p[1:] for p in ANIMATION_PRESETS]  # (z, x, y) for filter
BROLL_COLORS = ("#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560", "#2d4059", "#3d5a80", "#ee6c4d")

ASSETS_PROMPT = """You are an elite direct-response creative strategist for Meta/TikTok.

Create UGC-style ad assets for the product below.

Return ONLY valid JSON (no markdown, no commentary).

Schema:
{{
  "hooks": ["..."],
  "voiceover": "...",
  "text_overlays": ["..."],
  "broll_shotlist": [
    {{"time": "0-3s", "shot": "..."}},
    {{"time": "3-10s", "shot": "...", "video_url": "optional URL for 2-3s clip", "video_path": "optional local path"}}
  ],
  "cta": "..."
}}

Rules:
- hooks: 8 options, each <= 12 words, punchy, no emojis.
- voiceover: 35–55 seconds spoken (~95–130 words), conversational, specific.
- Mention the product name exactly once in voiceover.
- text_overlays: 6 short overlays (2–6 words), not redundant with voiceover.
- broll_shotlist: 8–10 shots, vertical-friendly. Each "shot" is a clear, visual search phrase for stock photos. Optional "video_url" or "video_path" for a 2–3s video clip (used instead of image for that slot).
- cta: 1 sentence, direct, no hype.

Inputs:
Product: {product_name}
Description: {product_description}
Target audience: {target_audience}
"""

_PEXELS_STOPWORDS = frozenset({
    "the", "a", "an", "of", "on", "with", "to", "into", "and", "or", "for", "in", "at", "by",
    "as", "from", "that", "this", "it", "is", "are", "was", "were", "been", "being", "their",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "shall", "can", "need", "used", "what", "which", "when", "where", "who", "how",
})

# Visual terms that improve Pexels results (order matters for relevance)
# These are intentionally specific, multi-word, and UGC-style to reduce random stock matches.
_VISUAL_TERMS = (
    # people
    "young woman", "young man", "person using phone", "person holding smartphone",
    "person looking at phone", "smiling person", "portrait selfie style",

    # hands / device interaction
    "hand holding smartphone", "close up phone in hand", "close up mobile screen",
    "scrolling phone", "typing on smartphone", "tapping phone screen",

    # phone / app usage
    "smartphone app interface", "mobile app screen", "checking phone",
    "using mobile app", "reading phone screen",

    # ugc / social style
    "selfie camera angle", "talking to camera", "review style video",
    "testimonial style", "casual lifestyle shot",

    # environments
    "at home", "on couch using phone", "sitting at desk with phone",
    "bedroom phone usage", "office phone usage",

    # visual quality
    "vertical photo", "natural light", "real lifestyle photography",
)

_CONTEXT_TERMS = (
    "living room", "bedroom", "home interior",
    "office workspace", "desk setup",
    "cafe environment", "coffee shop",
    "outdoors lifestyle", "city street",
    "night phone usage", "evening indoor lighting",
)


def _ensure_workdir():
    WORKDIR.mkdir(parents=True, exist_ok=True)
    (WORKDIR / "broll").mkdir(parents=True, exist_ok=True)


def _cache_key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _cache_path(cache_dir: Path, subdir: str, key: str, ext: str) -> Path:
    p = cache_dir / subdir / f"{key}{ext}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def compute_scene_timeline(n_scenes: int) -> list[tuple[float, float]]:
    """Pre-calculate (start_sec, end_sec) per scene for single-pass render."""
    if n_scenes <= 0:
        return []
    step = BROLL_CLIP_DURATION - XFADE_DURATION
    return [(i * step, i * step + BROLL_CLIP_DURATION) for i in range(n_scenes)]


def _parse_loose_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return valid JSON.")
    return json.loads(text[start : end + 1])


# --- Step 1: Script generation (Ollama, same model as local) ---
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def _ollama_healthcheck(timeout_s: float = 2.0) -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=timeout_s)
        return r.status_code == 200
    except requests.RequestException:
        return False


def ensure_ollama_running() -> bool:
    """Ensure Ollama is running in Colab. Returns True if ready."""
    if _ollama_healthcheck():
        return True
    # Try to start ollama serve (must be installed via notebook cell)
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(30):
            time.sleep(1)
            if _ollama_healthcheck():
                return True
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return False


def generate_assets_ollama(
    product_name: str,
    product_description: str,
    target_audience: str,
    model: str = OLLAMA_MODEL,
) -> dict:
    """Generate ad assets JSON using Ollama (same model as local pipeline). No API key needed."""
    if not ensure_ollama_running():
        raise RuntimeError(
            "Ollama not running. Run the 'Install Ollama' cell first, then ollama pull "
            f"{model} before running the pipeline."
        )
    prompt = ASSETS_PROMPT.format(
        product_name=product_name,
        product_description=product_description,
        target_audience=target_audience,
    )
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    try:
        r = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error ({r.status_code}): {r.text[:500]}")
    data = r.json()
    out = (data.get("response") or "").strip()
    if not out:
        raise RuntimeError("Ollama returned empty response.")
    return _parse_loose_json(out)


# --- Step 2: TTS (gTTS) ---
def generate_voice_gtts(text: str, output_wav: Path) -> None:
    """Generate voiceover with gTTS + convert to WAV via ffmpeg."""
    try:
        from gtts import gTTS
    except ImportError:
        raise RuntimeError("Install: pip install gtts")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    mp3_path = output_wav.with_suffix(".mp3")
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(str(mp3_path))
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3_path), "-acodec", "pcm_s16le", "-ar", "24000", str(output_wav)],
        check=True,
        capture_output=True,
    )
    if mp3_path.exists():
        mp3_path.unlink()


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


# --- Step 3: Captions (faster-whisper, CPU, small model) ---
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


def _parse_srt_ts(ts: str) -> float:
    # "HH:MM:SS,mmm"
    hh, mm, rest = ts.split(":", 2)
    ss, ms = rest.split(",", 1)
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def _get_last_srt_end_seconds(srt_path: Path) -> float | None:
    try:
        lines = srt_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    last_end: float | None = None
    for line in lines:
        if "-->" not in line:
            continue
        try:
            end = line.split("-->", 1)[1].strip().split()[0]
            last_end = _parse_srt_ts(end)
        except Exception:
            continue
    return last_end


def _extend_last_srt_end_to(srt_path: Path, new_end_seconds: float) -> bool:
    """Extend the final subtitle end timestamp to new_end_seconds. Returns True if modified."""
    if new_end_seconds <= 0:
        return False
    try:
        lines = srt_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return False
    new_end_str = format_srt_ts(float(new_end_seconds))
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if "-->" not in line:
            continue
        try:
            start_raw, end_raw = [p.strip() for p in line.split("-->", 1)]
            start_ts = start_raw.split()[0]
            # preserve any trailing style/position after end timestamp
            end_parts = end_raw.split()
            trailing = " " + " ".join(end_parts[1:]) if len(end_parts) > 1 else ""
            lines[i] = f"{start_ts} --> {new_end_str}{trailing}"
            srt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return True
        except Exception:
            continue
    return False


def validate_and_fix_captions(audio_path: Path, srt_path: Path, model_name: str = WHISPER_MODEL) -> bool:
    """
    Ensure captions cover the full audio duration.
    If SRT ends early, try regenerating once; if still early, extend last subtitle end time.
    Returns True if captions file was modified.
    """
    audio_dur = float(wav_duration_seconds(audio_path))
    last_end = _get_last_srt_end_seconds(srt_path)
    if last_end is None:
        generate_captions(audio_path, srt_path, model_name=model_name)
        last_end = _get_last_srt_end_seconds(srt_path)
        if last_end is None:
            return False
        # fallthrough into comparison
    # Allow small rounding gaps; otherwise fix.
    if last_end < audio_dur - 0.2:
        # Try once more to regenerate (helps when last segment got cut off).
        generate_captions(audio_path, srt_path, model_name=model_name)
        last_end2 = _get_last_srt_end_seconds(srt_path) or last_end
        if last_end2 < audio_dur - 0.2:
            return _extend_last_srt_end_to(srt_path, audio_dur)
        return True
    return False


def generate_captions(audio_path: Path, srt_path: Path, model_name: str = WHISPER_MODEL) -> None:
    """Word-level SRT with faster-whisper. Uses CPU and int8 to save RAM."""
    from faster_whisper import WhisperModel
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
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
    del model
    gc.collect()


# --- Step 4: Pexels (improved queries for visual accuracy) ---
def _extract_product_keywords(product_name: str, product_description: str, max_words: int = 5) -> str:
    combined = f"{product_name} {product_description}".lower()
    for c in ".,;:!?\"'()[]-–—":
        combined = combined.replace(c, " ")
    words = [w for w in combined.split() if w and len(w) > 2]
    generic = _PEXELS_STOPWORDS | {"app", "product", "daily", "using", "made", "get", "now"}
    kept = [w for w in words if w not in generic]
    seen = set()
    unique = []
    for w in kept:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return " ".join(unique[:max_words]) if unique else ""


def _shot_to_pexels_query(shot_text: str, product_keywords: str = "", max_words: int = 12) -> str:
    """Build Pexels query: product context first, then visual/context terms from shot. Optimized for stock results."""
    if not shot_text or not shot_text.strip():
        base = "person smartphone portrait"
    else:
        text = shot_text.lower().strip()
        for c in ".,;:!?\"'()[]-–—":
            text = text.replace(c, " ")
        words = [w for w in text.split() if w and w not in _PEXELS_STOPWORDS and len(w) > 1]

        # Phrase matches (e.g. "close up", "before and after") work better than token-only matching.
        phrases = [p for p in _VISUAL_TERMS if " " in p and p in text]
        context = [c for c in _CONTEXT_TERMS if c in text]

        # Lightweight, context-aware UGC boosts.
        boosts: list[str] = []
        if "before" in words and "after" in words:
            boosts.append("before and after")
        if any(k in words for k in ("review", "rating", "testimonial")):
            boosts.append("testimonial")
        if any(k in words for k in ("unboxing", "package", "packaging")):
            boosts.append("unboxing")
        if any(k in words for k in ("demo", "tutorial", "how")):
            boosts.append("demo")

        # Token-level visual terms (single words)
        visual_tokens = [t for t in _VISUAL_TERMS if " " not in t and t in words]

        ordered = []
        for item in (phrases + context + boosts + visual_tokens):
            if item and item not in ordered:
                ordered.append(item)
        # Add remaining shot words (keeps specificity like "birth chart", "skincare", etc.)
        for w in words:
            if w not in ordered:
                ordered.append(w)

        # Ensure a subject is present to avoid abstract results.
        if not any(k in ordered for k in ("person", "woman", "man", "people")):
            ordered.insert(0, "person")
        base = " ".join(ordered[:max_words]) if ordered else "person smartphone"
    if product_keywords and product_keywords.strip():
        return f"{product_keywords.strip()} {base}".strip()
    return base or "person smartphone portrait"


def _shot_to_sd_prompt(shot_text: str, product_keywords: str = "") -> str:
    """
    Stable Diffusion prompt for high‑quality UGC‑style vertical B‑roll.
    Produces detailed, photographic prompts tuned for TikTok/UGC aesthetics.
    """
    base = (shot_text or "").strip()
    pk = (product_keywords or "").strip()

    # Core subject: try to keep it on-theme with phone/app + human.
    core_subject = "young person using smartphone"
    lower = base.lower()
    if any(k in lower for k in ["woman", "girl", "female"]):
        core_subject = "young woman using smartphone"
    elif any(k in lower for k in ["man", "boy", "male"]):
        core_subject = "young man using smartphone"

    # Style & quality descriptors
    style_tokens = [
        "ultra realistic",
        "highly detailed",
        "professional photography",
        "shallow depth of field",
        "soft natural light",
        "cinematic lighting",
        "shot on mirrorless camera",
        "sharp focus on subject",
    ]

    # UGC / social video framing
    ugc_tokens = [
        "vertical 9:16 framing",
        "TikTok style",
        "authentic casual lifestyle",
        "candid moment",
        "natural expression",
        "no posed stock photo look",
    ]

    # Map some common intents in the shot text to visual details
    intent_tokens: list[str] = []
    if "review" in lower or "testimonial" in lower:
        intent_tokens += ["talking to camera", "subtle smile", "expressive face"]
    if "before" in lower and "after" in lower:
        intent_tokens += ["before and after feel", "clear product benefit implied"]
    if "typing" in lower or "texting" in lower:
        intent_tokens += ["close up hands typing on phone", "phone screen glowing softly"]
    if "bedroom" in lower or "night" in lower:
        intent_tokens += ["cozy bedroom", "warm lamp light", "evening mood"]
    if "office" in lower or "desk" in lower:
        intent_tokens += ["minimal desk setup", "laptop and phone on desk", "daylight from window"]
    if "cafe" in lower or "coffee" in lower:
        intent_tokens += ["coffee shop interior", "bokeh lights in background"]

    # Product context (optional, but keep it short and visual)
    product_tokens: list[str] = []
    if pk:
        product_tokens.append(pk)
    if any(k in lower for k in ["astrology", "horoscope", "zodiac", "birth chart"]):
        product_tokens += [
            "astrology app UI on phone screen",
            "subtle zodiac icons on screen",
        ]

    # Negative prompt to avoid overlays + artifacts
    negative_prompt = (
        "no text, no watermark, no logo, no UI chrome, no borders, "
        "no captions, no heavy filters, no distortion"
    )

    pieces: list[str] = []
    if base:
        pieces.append(base)
    pieces.append(core_subject)
    if product_tokens:
        pieces.append(", ".join(product_tokens))
    pieces.append(", ".join(style_tokens))
    pieces.append(", ".join(ugc_tokens))
    if intent_tokens:
        pieces.append(", ".join(intent_tokens))
    pieces.append(negative_prompt)

    prompt = ", ".join([p.strip(" ,") for p in pieces if p and p.strip()])
    return prompt


def _sd_is_available() -> bool:
    try:
        import torch  # noqa: F401
        from diffusers import StableDiffusionPipeline  # noqa: F401

        return True
    except Exception:
        return False


def _simplify_sd_prompt(prompt: str) -> str:
    # Keep it short and strongly photographic for retry.
    base = (prompt or "").split("no text", 1)[0].strip(" ,")
    return base + ", vertical photo, natural light, candid, realistic, no text, no watermark"


def _sd_prompt_cache_path(cache_dir: Path | None, prompt: str) -> Path | None:
    if not cache_dir:
        return None
    key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    p = cache_dir / "sd" / f"{key}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _sd_service_main(model_id: str, in_q: Queue, out_q: Queue) -> None:
    """
    Long-lived Stable Diffusion worker: image-only generation.
    - Uses StableDiffusionPipeline (still images only; no video/frames).
    - Loads weights once; applies GPU optimizations.
    - Output: one PNG per prompt. Moving footage must come from video_url/video_path, not SD.
    """
    try:
        import torch
        from diffusers import StableDiffusionPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        out_q.put({"type": "ready", "device": device})

        while True:
            msg = in_q.get()
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "stop":
                return
            if msg.get("type") != "gen":
                continue
            req_id = msg.get("id")
            prompts = msg.get("prompts") or []
            try:
                # Image-only: StableDiffusionPipeline returns still images (no video/frames).
                res = pipe(
                    prompt=prompts,
                    num_inference_steps=int(msg.get("steps", 20)),
                    guidance_scale=float(msg.get("guidance", 6.5)),
                    height=int(msg.get("height", 1024)),
                    width=int(msg.get("width", 576)),
                )
                images = res.images or []  # list of PIL Images, one per prompt
                out = []
                for im in images:
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    out.append(buf.getvalue())
                # Ensure length matches prompts
                if len(out) < len(prompts):
                    out.extend([None] * (len(prompts) - len(out)))
                out_q.put({"type": "result", "id": req_id, "images": out})
            except Exception as e:
                out_q.put({"type": "result", "id": req_id, "images": [None] * len(prompts), "error": str(e)[:300]})
    except Exception as e:
        try:
            out_q.put({"type": "error", "error": str(e)[:300]})
        except Exception:
            pass


class StableDiffusionService:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._in: Queue = Queue()
        self._out: Queue = Queue()
        self._p: Process | None = None
        self.device: str | None = None

    def start(self, load_timeout_s: float = 900.0) -> bool:
        self._p = Process(target=_sd_service_main, args=(self.model_id, self._in, self._out))
        self._p.daemon = True
        self._p.start()
        t0 = time.time()
        while time.time() - t0 < load_timeout_s:
            try:
                msg = self._out.get(timeout=1.0)
            except Exception:
                continue
            if isinstance(msg, dict) and msg.get("type") == "ready":
                self.device = msg.get("device")
                return True
            if isinstance(msg, dict) and msg.get("type") == "error":
                print("Stable Diffusion load error:", msg.get("error"), flush=True)
                return False
        return False

    def stop(self) -> None:
        try:
            self._in.put({"type": "stop"})
        except Exception:
            pass
        if self._p and self._p.is_alive():
            try:
                self._p.terminate()
            except Exception:
                pass

    def generate_batch(self, prompts: list[str], timeout_s: float) -> list[bytes | None] | None:
        if not self._p or not self._p.is_alive():
            return None
        req_id = f"{time.time():.6f}"
        try:
            self._in.put({"type": "gen", "id": req_id, "prompts": prompts})
        except Exception:
            return None
        t0 = time.time()
        while time.time() - t0 < max(1.0, float(timeout_s)):
            try:
                msg = self._out.get(timeout=0.5)
            except Exception:
                continue
            if isinstance(msg, dict) and msg.get("type") == "result" and msg.get("id") == req_id:
                imgs = msg.get("images") or []
                return [(b if isinstance(b, (bytes, bytearray)) else None) for b in imgs]
        return None


def _fetch_pexels_photos(shot_text: str, api_key: str, product_keywords: str, page: int) -> list:
    import requests
    query = _shot_to_pexels_query(shot_text, product_keywords)
    q = quote_plus(query[:200])
    url = f"{PEXELS_API_URL}?query={q}&per_page={PEXELS_PER_QUERY}&orientation=portrait&page={max(1, page)}"
    try:
        r = requests.get(url, headers={"Authorization": api_key}, timeout=15)
        r.raise_for_status()
        return (r.json().get("photos") or [])
    except Exception:
        return []


def _rank_photo(photo: dict) -> tuple:
    w = photo.get("width") or (photo.get("src") or {}).get("width") or 0
    h = photo.get("height") or (photo.get("src") or {}).get("height") or 0
    orientation_ok = 1 if (h >= w * 0.9) else 0
    return (orientation_ok, w * h)


def _fetch_pexels_best(shot_text: str, api_key: str, product_keywords: str, page: int) -> bytes | None:
    photos = _fetch_pexels_photos(shot_text, api_key, product_keywords, page)
    if not photos:
        return None
    ranked = [(p, _rank_photo(p)) for p in photos]
    ranked.sort(key=lambda x: x[1], reverse=True)
    img_url = ranked[0][0].get("src", {}).get("large") or ranked[0][0].get("src", {}).get("original")
    if not img_url:
        return None
    try:
        r = requests.get(img_url, timeout=20)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def _fetch_unsplash_photos(shot_text: str, access_key: str, product_keywords: str, page: int) -> list:
    """Search Unsplash; same query logic as Pexels for consistency. Returns list of photo objects."""
    query = _shot_to_pexels_query(shot_text, product_keywords)
    q = quote_plus(query[:200])
    url = f"{UNSPLASH_API_URL}?query={q}&per_page={PEXELS_PER_QUERY}&orientation=portrait&page={max(1, page)}"
    try:
        r = requests.get(url, headers={"Authorization": f"Client-ID {access_key}"}, timeout=15)
        r.raise_for_status()
        return r.json().get("results") or []
    except Exception:
        return []


def _rank_unsplash_photo(photo: dict) -> tuple:
    """Prefer portrait and larger resolution for vertical B-roll."""
    w = photo.get("width") or 0
    h = photo.get("height") or 0
    orientation_ok = 1 if (h >= w * 0.9) else 0
    return (orientation_ok, w * h)


def _fetch_unsplash_best(shot_text: str, access_key: str, product_keywords: str, page: int) -> bytes | None:
    photos = _fetch_unsplash_photos(shot_text, access_key, product_keywords, page)
    if not photos:
        return None
    ranked = [(p, _rank_unsplash_photo(p)) for p in photos]
    ranked.sort(key=lambda x: x[1], reverse=True)
    urls = ranked[0][0].get("urls") or {}
    img_url = urls.get("regular") or urls.get("full") or urls.get("raw")
    if not img_url:
        return None
    try:
        r = requests.get(img_url, timeout=20)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def _fetch_pexels_videos(shot_text: str, api_key: str, product_keywords: str, per_page: int = 5) -> list:
    """Search Pexels for videos; returns list of video objects with video_files."""
    query = _shot_to_pexels_query(shot_text, product_keywords)
    q = quote_plus(query[:200])
    url = f"{PEXELS_VIDEOS_URL}?query={q}&per_page={per_page}"
    try:
        r = requests.get(url, headers={"Authorization": api_key}, timeout=15)
        r.raise_for_status()
        return r.json().get("videos") or []
    except Exception:
        return []


def _fetch_pexels_video_best(shot_text: str, api_key: str, product_keywords: str) -> str | None:
    """Return a single video file URL suitable for a short B-roll clip (prefer portrait/small duration)."""
    videos = _fetch_pexels_videos(shot_text, api_key, product_keywords)
    if not videos:
        return None
    target_w, target_h = BROLL_SIZE[0], BROLL_SIZE[1]
    best_url: str | None = None
    best_score = -1
    for v in videos:
        files = v.get("video_files") or []
        for f in files:
            link = f.get("link")
            if not link:
                continue
            w = f.get("width") or 0
            h = f.get("height") or 0
            if w < 400 or h < 400:
                continue
            # Prefer portrait or near-vertical for 9:16 B-roll
            portrait = 1 if h >= w else 0
            score = portrait * 10000 + min(w * h, target_w * target_h * 2)
            if score > best_score:
                best_score = score
                best_url = link
    return best_url


def _resize_crop_save(image_bytes: bytes, output_path: Path) -> bool:
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        target_w, target_h = BROLL_SIZE
        scale = max(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "PNG")
        return True
    except Exception:
        return False


def _placeholder_image(shot_text: str, output_path: Path, color_hex: str) -> None:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", BROLL_SIZE, color_hex)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
    words = shot_text.split()
    lines, current = [], []
    for w in words:
        current.append(w)
        if len(" ".join(current)) > 38:
            lines.append(" ".join(current))
            current = []
    if current:
        lines.append(" ".join(current))
    text_block = "\n".join(lines[:6])
    try:
        bbox = draw.textbbox((0, 0), text_block, font=font)
    except AttributeError:
        bbox = (0, 0, 400, 120)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (BROLL_SIZE[0] - tw) // 2
    y = (BROLL_SIZE[1] - th) // 2
    draw.text((x, y), text_block, fill="white", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")


def _get_shotlist(assets: dict) -> list[dict]:
    """Return list of shot items. Each may have time, shot (image), and/or video_url / video_path (short clip)."""
    shotlist = assets.get("broll_shotlist") or []
    out = []
    for item in shotlist:
        if not isinstance(item, dict):
            continue
        shot = str(item.get("shot") or "").strip()
        video_url = str(item.get("video_url") or "").strip() or None
        video_path = str(item.get("video_path") or "").strip() or None
        if not shot and not video_url and not video_path:
            continue
        out.append({
            "time": str(item.get("time") or "").strip(),
            "shot": shot or "",
            "video_url": video_url,
            "video_path": video_path,
        })
    return out


def _placeholder_only_one(i: int, shot_text: str, broll_dir: Path, cache_dir: Path | None, product_kw: str, use_broll_cache: bool) -> Path:
    """Write placeholder image for one scene and log source."""
    out_path = broll_dir / f"scene_{i:03d}.png"
    _placeholder_image(shot_text, out_path, BROLL_COLORS[i % len(BROLL_COLORS)])
    if use_broll_cache and cache_dir:
        cache_key = _cache_key(product_kw, shot_text, str(i))
        shutil.copy2(out_path, _cache_path(cache_dir, "images", cache_key, ".png"))
    print(f"B-roll {i+1}: placeholder -> {out_path.name}", flush=True)
    return out_path


def generate_broll_images(
    assets: dict,
    product_name: str,
    product_description: str,
    cache_dir: Path | None = None,
) -> list[Path | None]:
    """
    Fetch/preprocess still images for image scenes; video scenes get None.
    Flow: Unsplash → Pexels → Stable Diffusion → placeholder. Video slots use video_url/video_path or Pexels Video API.
    SD is image-only; moving footage comes from video URLs or Pexels video search.
    """
    # Read at runtime so env set in notebook is respected (default off = overwrite every time)
    use_broll_cache = os.environ.get("UGC_BROLL_USE_CACHE", "0").strip().lower() in ("1", "true", "yes")
    shotlist = _get_shotlist(assets)
    if not shotlist:
        shotlist = [{"time": "0-0s", "shot": "person using smartphone", "video_url": None, "video_path": None}]
    if MAX_BROLL_SHOTS > 0:
        shotlist = shotlist[:MAX_BROLL_SHOTS]
    api_key = (os.environ.get("PEXELS_API_KEY") or "").strip()
    unsplash_key = (os.environ.get("UNSPLASH_ACCESS_KEY") or "").strip()
    product_kw = _extract_product_keywords(product_name, product_description)
    broll_dir = WORKDIR / "broll"
    broll_dir.mkdir(parents=True, exist_ok=True)
    if not use_broll_cache:
        for p in broll_dir.glob("scene_*.png"):
            try:
                p.unlink()
            except Exception:
                pass
    sd_available = _sd_is_available()
    print(
        f"B-roll generation: Unsplash={'on' if unsplash_key else 'off'} Pexels={'on' if api_key else 'off'} SD_ENABLED={SD_ENABLED} sd_available={sd_available} cache={'on' if use_broll_cache else 'off (overwrite)'}",
        flush=True,
    )

    # Video scenes: no image; leave out_paths[i] = None
    out_paths: list[Path | None] = [None] * len(shotlist)
    pending: list[int] = []
    for i, item in enumerate(shotlist):
        if item.get("video_url") or item.get("video_path"):
            continue  # clip will be built from video in build_broll_clips
        out_path = broll_dir / f"scene_{i:03d}.png"
        shot_text = item.get("shot") or ""
        if not shot_text:
            continue
        cache_key = _cache_key(product_kw, shot_text, str(i))
        if use_broll_cache and cache_dir:
            cached = _cache_path(cache_dir, "images", cache_key, ".png")
            if cached.exists():
                shutil.copy2(cached, out_path)
                print(f"B-roll {i+1}: cache -> {out_path.name}", flush=True)
                out_paths[i] = out_path
                continue
        pending.append(i)

    # Stage B: Unsplash first, then Pexels (per-scene); remaining go to SD
    sd_pending: list[int] = []
    for i in pending:
        out_path = broll_dir / f"scene_{i:03d}.png"
        shot_text = shotlist[i]["shot"]
        raw: bytes | None = None
        source = ""
        if unsplash_key:
            raw = _fetch_unsplash_best(shot_text, unsplash_key, product_kw, page=i + 1)
            if raw:
                source = "Unsplash"
        if not raw and api_key:
            raw = _fetch_pexels_best(shot_text, api_key, product_kw, page=i + 1)
            if raw:
                source = "Pexels fallback"
        if raw and _resize_crop_save(raw, out_path):
            if use_broll_cache and cache_dir:
                cache_key = _cache_key(product_kw, shot_text, str(i))
                shutil.copy2(out_path, _cache_path(cache_dir, "images", cache_key, ".png"))
            print(f"B-roll {i+1}: {source} -> {out_path.name}", flush=True)
            out_paths[i] = out_path
        else:
            sd_pending.append(i)

    # Stage C: Stable Diffusion for scenes that got no image from Unsplash/Pexels
    failed_for_fallback: set[int] = set(sd_pending)
    if SD_ENABLED and sd_available and sd_pending:
        service = StableDiffusionService(SD_MODEL_ID)
        print("Loading Stable Diffusion model...", flush=True)
        loaded = service.start(load_timeout_s=900.0)
        print(f"Stable Diffusion ready: {loaded} device={service.device}", flush=True)
        if loaded:
            try:
                # First pass prompts
                prompts = {i: _shot_to_sd_prompt(shotlist[i]["shot"], product_keywords=product_kw) for i in sd_pending}

                # Use prompt-cache when available (skipped when use_broll_cache is False)
                to_gen: list[int] = []
                for i in sd_pending:
                    if use_broll_cache:
                        cp = _sd_prompt_cache_path(cache_dir, prompts[i])
                        if cp and cp.exists():
                            png = cp.read_bytes()
                            out_path = broll_dir / f"scene_{i:03d}.png"
                            if _resize_crop_save(png, out_path):
                                if cache_dir:
                                    shot_text = shotlist[i]["shot"]
                                    cache_key = _cache_key(product_kw, shot_text, str(i))
                                    shutil.copy2(out_path, _cache_path(cache_dir, "images", cache_key, ".png"))
                                print(f"B-roll {i+1}: Stable Diffusion (prompt-cache) -> {out_path.name}", flush=True)
                                out_paths[i] = out_path
                                failed_for_fallback.discard(i)
                                continue
                    to_gen.append(i)

                def _gen_indices(idxs: list[int], use_simplified: bool = False) -> set[int]:
                    still_missing: set[int] = set()
                    bs = max(1, int(SD_BATCH_SIZE))
                    for j in range(0, len(idxs), bs):
                        batch_idxs = idxs[j : j + bs]
                        batch_prompts = []
                        for ii in batch_idxs:
                            p = prompts[ii]
                            batch_prompts.append(_simplify_sd_prompt(p) if use_simplified else p)
                        # Timeout starts after model is loaded (service is ready here).
                        timeout = float(SD_TIMEOUT_S) * max(1, len(batch_prompts))
                        imgs = service.generate_batch(batch_prompts, timeout_s=timeout)
                        if imgs is None:
                            # treat entire batch as failed
                            still_missing.update(batch_idxs)
                            continue
                        for ii, png in zip(batch_idxs, imgs):
                            if not png:
                                still_missing.add(ii)
                                continue
                            out_path = broll_dir / f"scene_{ii:03d}.png"
                            if _resize_crop_save(png, out_path):
                                # prompt-cache write when use_broll_cache (for the prompt variant used)
                                if use_broll_cache:
                                    used_prompt = batch_prompts[batch_idxs.index(ii)]
                                    cp = _sd_prompt_cache_path(cache_dir, used_prompt)
                                    if cp and not cp.exists():
                                        try:
                                            cp.write_bytes(png)
                                        except Exception:
                                            pass
                                    if cache_dir:
                                        shot_text = shotlist[ii]["shot"]
                                        cache_key = _cache_key(product_kw, shot_text, str(ii))
                                        shutil.copy2(out_path, _cache_path(cache_dir, "images", cache_key, ".png"))
                                tag = "Stable Diffusion (retry)" if use_simplified else "Stable Diffusion"
                                print(f"B-roll {ii+1}: {tag} -> {out_path.name}", flush=True)
                                out_paths[ii] = out_path
                                failed_for_fallback.discard(ii)
                            else:
                                still_missing.add(ii)
                    return still_missing

                missing = _gen_indices(to_gen, use_simplified=False)
                if missing:
                    # Retry with simplified prompts
                    missing2 = _gen_indices(sorted(missing), use_simplified=True)
                    for m in missing2:
                        failed_for_fallback.add(m)
            finally:
                service.stop()

    # Stage D: Placeholder for any scene that still has no image (Unsplash/Pexels/SD all skipped or failed)
    remaining = sorted([i for i in failed_for_fallback if out_paths[i] is None])
    for i in remaining:
        out_paths[i] = _placeholder_only_one(
            i, shotlist[i]["shot"], broll_dir, cache_dir, product_kw, use_broll_cache
        )

    return out_paths


# --- Step 5 & 6: Two-stage FFmpeg (per-image zoompan clips, then xfade assembly) ---
# Stage 1: one short animated clip per B-roll image (zoompan). Stage 2: concat clips with
# xfade transitions and burn subtitles. Avoids one huge filter_complex; same animations.
def _ffmpeg_subtitles_filter(srt_path: Path) -> str:
    path_str = str(srt_path.resolve()).replace("'", "'\\''")
    return f"subtitles='{path_str}'"


def _detect_gpu_encoder() -> str:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and "h264_nvenc" in (out.stdout or ""):
            return "h264_nvenc"
    except Exception:
        pass
    return "libx264"


def _image_to_animated_clip(
    image_path: Path,
    output_mp4: Path,
    duration_sec: float = BROLL_CLIP_DURATION,
    clip_index: int = 0,
) -> None:
    """Stage 1: Render one image as a short vertical clip with Ken Burns (zoompan). One preset per clip."""
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    frames = max(1, int(round(duration_sec * BROLL_FPS)))
    z, x, y = ZOOM_PRESETS[clip_index % len(ZOOM_PRESETS)]
    zoompan = (
        f"zoompan=z='{z}':x='{x}':y='{y}':"
        f"d={frames}:s={BROLL_SIZE[0]}x{BROLL_SIZE[1]}:fps={BROLL_FPS}"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-loop", "1", "-i", str(image_path),
            "-vf", zoompan,
            "-t", f"{duration_sec:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(BROLL_FPS),
            str(output_mp4),
        ],
        check=True,
        capture_output=True,
    )


def _video_to_clip(
    url_or_path: str,
    output_mp4: Path,
    duration_sec: float = BROLL_CLIP_DURATION,
) -> bool:
    """
    Turn a short video (URL or local path) into a vertical clip: trim to duration_sec, scale/crop to BROLL_SIZE, 30fps.
    Returns True on success, False on failure (caller can fall back to placeholder).
    """
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    w, h = BROLL_SIZE[0], BROLL_SIZE[1]
    # scale to cover then center-crop to exact size
    vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}:(iw-{w})/2:(ih-{h})/2"
    input_path: Path | None = None
    if url_or_path.startswith(("http://", "https://")):
        try:
            r = requests.get(url_or_path, timeout=30)
            r.raise_for_status()
            input_path = output_mp4.with_suffix(".tmp_video")
            input_path.write_bytes(r.content)
        except Exception:
            return False
    else:
        input_path = Path(url_or_path)
        if not input_path.exists():
            return False
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(input_path),
                "-t", f"{duration_sec:.3f}",
                "-vf", vf,
                "-r", str(BROLL_FPS),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(output_mp4),
            ],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        if input_path and input_path != Path(url_or_path) and input_path.exists():
            try:
                input_path.unlink()
            except Exception:
                pass


def _black_clip(out_mp4: Path) -> None:
    """Generate a black placeholder clip so assembly has N clips."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s={BROLL_SIZE[0]}x{BROLL_SIZE[1]}:r={BROLL_FPS}",
            "-t", f"{BROLL_CLIP_DURATION:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4),
        ],
        check=True,
        capture_output=True,
    )


def build_broll_clips(
    shotlist: list[dict],
    scene_assets: list[Path | None],
    clips_dir: Path,
    product_kw: str = "",
    pexels_api_key: str | None = None,
) -> list[Path]:
    """
    Build one clip per shotlist item: from image (zoompan) or from short video.
    Video source: shotlist video_url/video_path, or Pexels Video API if no URL given.
    scene_assets[i] is Path for image scenes, None for video scenes. Order preserved.
    """
    if not shotlist or len(shotlist) != len(scene_assets):
        return []
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[Path] = []
    api_key = (pexels_api_key or "").strip() or None
    for i in range(len(shotlist)):
        out_mp4 = clips_dir / f"clip_{i:03d}.mp4"
        if scene_assets[i] is not None:
            _image_to_animated_clip(scene_assets[i], out_mp4, clip_index=i)
            clip_paths.append(out_mp4)
        elif shotlist[i].get("video_url") or shotlist[i].get("video_path"):
            url_or_path = shotlist[i].get("video_url") or shotlist[i].get("video_path") or ""
            if _video_to_clip(url_or_path, out_mp4):
                clip_paths.append(out_mp4)
                print(f"B-roll clip {i+1}: video (URL/path) -> {out_mp4.name}", flush=True)
            else:
                _black_clip(out_mp4)
                clip_paths.append(out_mp4)
        else:
            # No explicit URL: try Pexels Video API for this slot
            shot_text = (shotlist[i].get("shot") or "").strip()
            if api_key and shot_text:
                video_url = _fetch_pexels_video_best(shot_text, api_key, product_kw or "")
                if video_url and _video_to_clip(video_url, out_mp4):
                    clip_paths.append(out_mp4)
                    print(f"B-roll clip {i+1}: Pexels video -> {out_mp4.name}", flush=True)
                else:
                    _black_clip(out_mp4)
                    clip_paths.append(out_mp4)
            else:
                _black_clip(out_mp4)
                clip_paths.append(out_mp4)
    return clip_paths


def broll_images_to_clips(image_paths: list[Path], clips_dir: Path) -> list[Path]:
    """Stage 1: Generate one animated clip per B-roll image (zoompan, same presets as before)."""
    if not image_paths:
        return []
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_paths = []
    for i, img_path in enumerate(image_paths):
        out_mp4 = clips_dir / f"clip_{i:03d}.mp4"
        _image_to_animated_clip(img_path, out_mp4, clip_index=i)
        clip_paths.append(out_mp4)
    return clip_paths


def _assemble_clips_with_xfade(
    clip_paths: list[Path],
    audio_wav: Path,
    srt_path: Path,
    out_mp4: Path,
) -> None:
    """Stage 2: Concat clips with xfade transitions, add voiceover, burn captions. Uses libx264 for Colab."""
    if not clip_paths:
        _black_video_fallback(audio_wav, srt_path, out_mp4)
        return
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    n = len(clip_paths)
    dur = max(1.0, wav_duration_seconds(audio_wav) + 0.25)
    fd = XFADE_DURATION
    offset_step = BROLL_CLIP_DURATION - fd
    filter_parts = []
    for i in range(n - 1):
        left = f"[v{i}]" if i > 0 else "[0:v]"
        right = f"[{i+1}:v]"
        offset = offset_step * (i + 1)
        out = f"[v{i+1}]"
        trans = XFADE_TRANSITIONS[i % len(XFADE_TRANSITIONS)]
        filter_parts.append(f"{left}{right}xfade=transition={trans}:duration={fd}:offset={offset:.2f}{out}")
    filter_chain = ";".join(filter_parts)
    last_v = f"[v{n-1}]" if n > 1 else "[0:v]"
    subs = _ffmpeg_subtitles_filter(srt_path)
    # Pad/freeze the final frame first so visuals extend to the full audio duration,
    # then burn subtitles on top so captions remain time-accurate through the padded tail.
    filter_chain += ";" + last_v + "tpad=stop_mode=clone:stop_duration=3600[vpad]"
    filter_chain += ";[vpad]" + subs + "[vout]"
    # Colab does not provide NVENC for ffmpeg; always use libx264.
    encoder = "libx264"
    cmd = ["ffmpeg", "-y"]
    for c in clip_paths:
        cmd.extend(["-i", str(c)])
    cmd.extend(["-i", str(audio_wav)])
    cmd.extend([
        "-filter_complex", filter_chain,
        "-map", "[vout]", "-map", f"{n}:a:0",
        "-t", f"{dur:.3f}",
        "-c:v", encoder, "-pix_fmt", "yuv420p",
    ])
    # Don't use -shortest: we explicitly target the full audio duration via -t.
    cmd.extend(["-c:a", "aac", str(out_mp4)])
    subprocess.run(cmd, check=True)


def _black_video_fallback(audio_wav: Path, srt_path: Path, out_mp4: Path) -> None:
    dur = max(1.0, wav_duration_seconds(audio_wav) + 0.25)
    subs = _ffmpeg_subtitles_filter(srt_path)
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:r=30",
        "-i", str(audio_wav), "-t", f"{dur:.3f}",
        "-vf", subs, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", str(out_mp4),
    ], check=True)


def run_full_pipeline(
    product_name: str,
    product_description: str,
    target_audience: str,
    whisper_model: str = WHISPER_MODEL,
    ollama_model: str = OLLAMA_MODEL,
    start_from: str = "script",
) -> dict:
    """
    Run the full UGC ad pipeline in Colab.
    Returns dict with paths: assets, voiceover_txt, voiceover_wav, captions_srt, final_mp4, broll_dir.
    Requires: Ollama installed + model pulled (run Install Ollama cell first).
    Optional: UNSPLASH_ACCESS_KEY and/or PEXELS_API_KEY for B-roll images; PEXELS_API_KEY also used for video search. Otherwise placeholders.
    Uses caching for audio, captions, and images when rerunning with same inputs.
    """
    _ensure_workdir()
    cache_dir = WORKDIR / CACHE_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)

    stages = ("script", "voice", "captions", "broll", "clips", "assemble")
    if start_from not in stages:
        raise ValueError(f"start_from must be one of {stages}, got: {start_from!r}")
    start_idx = stages.index(start_from)

    def _require_exists(p: Path, label: str) -> Path:
        if not p.exists():
            raise FileNotFoundError(f"Missing {label} at {p}. Run from an earlier stage.")
        return p

    paths: dict[str, Path] = {}
    assets: dict | None = None
    voiceover: str | None = None
    assets_path = WORKDIR / ASSETS_JSON_FILENAME
    voiceover_path = WORKDIR / VOICEOVER_TXT_FILENAME
    audio_path = WORKDIR / VOICEOVER_WAV_FILENAME
    captions_path = WORKDIR / CAPTIONS_SRT_FILENAME

    # 1. Script (Ollama) OR load previous assets/voiceover
    if start_idx <= 0:
        print("1/6 Generating script + assets (Ollama)...")
        assets = generate_assets_ollama(
            product_name, product_description, target_audience, model=ollama_model
        )
        assets_path.write_text(json.dumps(assets, indent=2), encoding="utf-8")
        voiceover = (assets.get("voiceover") or "").strip()
        if not voiceover:
            raise RuntimeError("Generated assets missing 'voiceover'.")
        voiceover_path.write_text(voiceover + "\n", encoding="utf-8")
    else:
        assets_path = _require_exists(assets_path, f"assets JSON ({ASSETS_JSON_FILENAME})")
        voiceover_path = _require_exists(voiceover_path, f"voiceover text ({VOICEOVER_TXT_FILENAME})")
        assets = json.loads(assets_path.read_text(encoding="utf-8"))
        voiceover = voiceover_path.read_text(encoding="utf-8").strip()
        if not voiceover:
            raise RuntimeError("voiceover.txt is empty. Run from 'script'.")

    paths["assets"] = assets_path
    paths["voiceover_txt"] = voiceover_path

    # 2. Voice (gTTS) OR reuse existing file
    key_vo = _cache_key(voiceover)
    if start_idx <= 1:
        print("2/6 Generating voiceover (gTTS)...")
        cached_audio = _cache_path(cache_dir, "audio", key_vo, ".wav")
        if cached_audio.exists():
            shutil.copy2(cached_audio, audio_path)
        else:
            generate_voice_gtts(voiceover, audio_path)
            shutil.copy2(audio_path, cached_audio)
    else:
        _require_exists(audio_path, f"voiceover audio ({VOICEOVER_WAV_FILENAME})")
    paths["voiceover_wav"] = audio_path

    # 3. Captions (faster-whisper) OR reuse existing file
    if start_idx <= 2:
        print("3/6 Generating captions (faster-whisper)...")
        cached_srt = _cache_path(cache_dir, "captions", key_vo, ".srt")
        if cached_srt.exists():
            shutil.copy2(cached_srt, captions_path)
        else:
            generate_captions(audio_path, captions_path, model_name=whisper_model)
            shutil.copy2(captions_path, cached_srt)
        # Validate captions cover the full audio; fix if needed and refresh cache.
        if validate_and_fix_captions(audio_path, captions_path, model_name=whisper_model):
            shutil.copy2(captions_path, cached_srt)
        gc.collect()
    else:
        _require_exists(captions_path, f"captions SRT ({CAPTIONS_SRT_FILENAME})")
    paths["captions_srt"] = captions_path

    # 4. B-roll images (and video slots) OR reuse existing images from broll dir
    broll_dir = WORKDIR / "broll"
    scene_assets: list[Path | None] | list[Path]
    shotlist = _get_shotlist(assets or {})
    if start_idx <= 3:
        print("4/6 Generating B-roll images (Unsplash → Pexels → SD → placeholder)...")
        scene_assets = generate_broll_images(
            assets or {}, product_name, product_description, cache_dir=cache_dir
        )
    else:
        _require_exists(broll_dir, "B-roll directory (broll/)")
        scene_assets = sorted(broll_dir.glob("scene_*.png"))
        if not scene_assets:
            raise FileNotFoundError("No broll images found (expected broll/scene_*.png). Run from 'broll'.")
    paths["broll_dir"] = broll_dir

    # 5. Build clips: image → zoompan; video_url/video_path → trim/resize (2–3s). Or reuse existing clips.
    clips_dir = WORKDIR / "clips"
    clip_paths: list[Path]
    if start_idx <= 4:
        print("5/6 Generating animated clips (images + optional video clips)...")
        if start_idx <= 3 and shotlist and len(shotlist) == len(scene_assets):
            _product_kw = _extract_product_keywords(product_name, product_description)
            _pexels_key = (os.environ.get("PEXELS_API_KEY") or "").strip() or None
            clip_paths = build_broll_clips(shotlist, scene_assets, clips_dir, product_kw=_product_kw, pexels_api_key=_pexels_key)
        else:
            clip_paths = broll_images_to_clips(
                [p for p in scene_assets if p is not None],
                clips_dir,
            )
    else:
        _require_exists(clips_dir, "clips directory (clips/)")
        clip_paths = sorted(clips_dir.glob("clip_*.mp4"))
        if not clip_paths:
            raise FileNotFoundError("No clips found (expected clips/clip_*.mp4). Run from 'clips'.")
    paths["clips_dir"] = clips_dir

    # 6. Assemble final video
    print("6/6 Assembling final video (xfade + captions)...")
    final_path = WORKDIR / FINAL_MP4_FILENAME
    if clip_paths:
        _assemble_clips_with_xfade(clip_paths, audio_path, captions_path, final_path)
    else:
        _black_video_fallback(audio_path, captions_path, final_path)
    paths["final_mp4"] = final_path

    print("Done. final.mp4:", final_path)
    return paths
