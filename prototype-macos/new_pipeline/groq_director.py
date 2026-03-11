from __future__ import annotations

import json
import time
from pathlib import Path

from groq import Groq

from utils import list_png_files, list_symbol_files, safe_json_dumps


SYSTEM_PROMPT = """
You are a professional UGC ad director. You receive an app config and produce a complete video plan.
You adapt everything — tone, scenes, backgrounds, symbols, text, color grades — entirely based on the app niche and target audience provided. You never assume a niche or hardcode any content.

OUTPUT FORMAT: Return ONLY raw valid JSON. No preamble. No markdown. No code fences. No explanation. Just the JSON object starting with { and ending with }.

SCENE TYPES:
- Type A: Person holding phone. App screenshot will be composited onto screen separately by code. Set flux_prompt to null for type A.
- Type B: Atmospheric or thematic scene matching the niche. Symbols and text overlaid separately by code.
- Type C: Pure emotion or lifestyle shot. No text overlay needed. Just atmosphere.

STRICT FLUX PROMPT RULES (apply to every scene, every niche, no exceptions):
- End every flux_prompt with: cinematic lighting, photorealistic, 8k, no text, no symbols, no writing, no letters, no signs
- Never include in any flux_prompt: open books, scrolls, calendars, newspapers, chalkboards, zodiac wheels, charts, any writing surface, any screen content, any UI elements
- Use only atmospheric, text-free, symbol-free backgrounds that match the niche mood
- Fitness niche: gym equipment, outdoor run, green smoothie, energy
- Astrology niche: cosmic space, candlelight, dark velvet, smoke, stars
- Finance niche: clean modern office, confident person, city lights, bokeh
- Food niche: fresh ingredients, warm kitchen lighting, steam, colors
- Education niche: clean desk, warm study light, focused person (no visible text on anything)
- For any other niche: use atmospheric backgrounds that match the emotional tone

SYMBOL AND OVERLAY RULES:
- If the niche has associated symbols, icons, or visual elements, specify them in the overlay field as symbol_file
- The code will render these from real PNG asset files — do not ask FLUX to generate them
- Pick symbol_file names that match what would logically exist in the niche subfolder

CAMERA MOVEMENT — pick one per scene, never repeat the same movement twice in a row:
slow_zoom_in, slow_zoom_out, pan_left, pan_right, pan_top_bottom, diagonal_drift

COLOR GRADE — pick based on niche mood and scene emotion:
warm_golden: spiritual, food, lifestyle, warm emotions
cool_mystical: tech, astrology, wellness, mystery
dark_cinematic: finance, premium, drama, intensity
soft_warm: family, education, health, trust

TRANSITION — pick one per scene:
crossfade, fadeblack, wipeleft

TEXT RULES:
- All overlay text must be in the language specified by ad_language
- Text must match the emotional tone of the scene
- Keep text short: maximum 6 words per overlay
- text_color must contrast with scene: dark scene = white or gold, light scene = dark color
"""


def fix_duplicate_movements(plan_dict: dict) -> dict:
    movements = [
        "slow_zoom_in",
        "slow_zoom_out",
        "pan_left",
        "pan_right",
        "pan_top_bottom",
        "diagonal_drift",
    ]
    scenes = plan_dict.get("scenes") or []
    for i in range(1, len(scenes)):
        prev = (scenes[i - 1] or {}).get("camera_movement")
        curr = (scenes[i] or {}).get("camera_movement")
        if prev and curr and curr == prev:
            alternatives = [m for m in movements if m != prev]
            scenes[i]["camera_movement"] = alternatives[i % len(alternatives)]
    return plan_dict


def _strip_fences(raw: str) -> str:
    s = (raw or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1].strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return s.strip()


def normalize_plan(raw: dict, config: dict) -> dict:
    for scene in raw.get("scenes", []):
        # Fix scene type field name
        if "scene_type" in scene and "type" not in scene:
            scene["type"] = scene.pop("scene_type")

        # Normalize scene type values (e.g. "Type A" -> "A")
        if "type" in scene and isinstance(scene.get("type"), str):
            t = scene["type"].strip()
            if t.lower().startswith("type "):
                t = t[5:].strip()
            if t.upper() in ("A", "B", "C"):
                scene["type"] = t.upper()

        # Fix duration field name
        if "duration" in scene and "duration_sec" not in scene:
            scene["duration_sec"] = scene.pop("duration")

        # Fix transition field name
        if "transition" in scene and "transition_out" not in scene:
            scene["transition_out"] = scene.pop("transition")

        # Add missing id if not present
        if "id" not in scene:
            scene["id"] = raw["scenes"].index(scene) + 1

        # Add missing voiceover_line if not present
        if "voiceover_line" not in scene:
            scene["voiceover_line"] = ""

    if "app_name" not in raw:
        raw["app_name"] = config.get("app_name", "")
    if "app_niche" not in raw:
        raw["app_niche"] = config.get("app_niche", "")
    if "video_topic" not in raw:
        raw["video_topic"] = raw.get("topic", "Ad Video")
    if "dominant_mood" not in raw:
        raw["dominant_mood"] = "energetic"
    if "voiceover_full" not in raw:
        raw["voiceover_full"] = " ".join(
            s.get("voiceover_line", "")
            for s in raw.get("scenes", [])
        )

    return raw


def _resolve_symbol_file(plan_dict: dict, available_symbols: set[str]) -> dict:
    # If requested symbol not found, try: keep if exists, else null.
    for scene in plan_dict.get("scenes") or []:
        overlay = scene.get("overlay") or {}
        sym = overlay.get("symbol_file")
        if sym and sym not in available_symbols:
            overlay["symbol_file"] = None
            scene["overlay"] = overlay
    return plan_dict


def _resolve_screenshot_file(plan_dict: dict, available_screens: list[str]) -> dict:
    fallback = available_screens[0] if available_screens else None
    for scene in plan_dict.get("scenes") or []:
        overlay = scene.get("overlay") or {}
        if overlay.get("type") == "app_screenshot":
            ss = overlay.get("screenshot_file")
            if not ss or ss not in available_screens:
                overlay["screenshot_file"] = fallback
                scene["overlay"] = overlay
    return plan_dict


def generate_scene_plan(config):
    """
    Groq director: returns a validated plan dict with exactly 8 scenes.
    Edge cases:
    - invalid JSON → 3 retries with backoff
    - markdown fences → stripped
    - rate limit 429 → sleep 10s and retry once (within attempts)
    """
    api_key = (config.get("groq_api_key") or "").strip()
    if not api_key:
        raise RuntimeError("Missing CONFIG['groq_api_key']. Add your Groq API key and rerun.")

    assets_base_path = Path(config["drive_base_path"])
    mockups = list_png_files(assets_base_path / "mockups")
    screenshots = list_png_files(assets_base_path / "screenshots")
    symbols = list_symbol_files(assets_base_path, str(config.get("app_niche") or "generic"))

    client = Groq(api_key=api_key)

    user_message = f"""
    App Name: {config['app_name']}
    App Niche: {config['app_niche']}
    App Description: {config['app_description']}
    Target Audience: {config['target_audience']}
    Ad Language: {config['ad_language']}
    Number of scenes needed: 8
    Total video duration: 30 seconds

    Available mockup files: {mockups}
    Available screenshot files: {screenshots}
    Available symbol files: {symbols}

    Generate a complete 8-scene video plan as JSON.
    """

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=4000,
                temperature=0.7,
            )
            raw = _strip_fences(response.choices[0].message.content or "")
            plan_dict = json.loads(raw)
            plan_dict = normalize_plan(plan_dict, config)
            scenes = plan_dict.get("scenes") or []
            if len(scenes) != 8:
                raise ValueError(f"Expected 8 scenes, got {len(scenes)}")
            return plan_dict
        except Exception as e:
            msg = str(e)
            print(f"Groq attempt {attempt+1} failed: {msg}", flush=True)
            last_err = e
            if "429" in msg or "rate" in msg.lower():
                time.sleep(10)
            # Spec: 3 retries with 2 second sleep between attempts on any failure.
            if attempt < 2:
                time.sleep(2)
            continue

    raise RuntimeError(f"Groq failed after 3 attempts: {last_err}")


def debug_print_plan(plan: dict) -> None:
    print(safe_json_dumps(plan), flush=True)

