from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageFont, ImageDraw


def apply_color_grade(image: Image.Image, grade: str) -> Image.Image:
    img = image.copy().convert("RGB")
    grades = {
        "warm_golden": {"brightness": 1.02, "contrast": 1.10, "saturation": 1.30, "r": 1.10, "g": 0.95, "b": 0.85},
        "cool_mystical": {"brightness": 0.97, "contrast": 1.15, "saturation": 1.20, "r": 0.90, "g": 0.95, "b": 1.15},
        "dark_cinematic": {"brightness": 0.95, "contrast": 1.20, "saturation": 0.85, "r": 1.00, "g": 0.95, "b": 0.90},
        "soft_warm": {"brightness": 1.05, "contrast": 1.05, "saturation": 1.10, "r": 1.05, "g": 1.02, "b": 0.92},
    }
    g = grades.get(grade, grades["soft_warm"])
    img = ImageEnhance.Brightness(img).enhance(g["brightness"])
    img = ImageEnhance.Contrast(img).enhance(g["contrast"])
    img = ImageEnhance.Color(img).enhance(g["saturation"])
    r, gr, b = img.split()
    r = r.point(lambda x: min(255, int(x * g["r"])))
    gr = gr.point(lambda x: min(255, int(x * g["g"])))
    b = b.point(lambda x: min(255, int(x * g["b"])))
    return Image.merge("RGB", (r, gr, b))


def detect_screen_corners(mockup_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(mockup_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = None
    best_area = 0.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 10000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best = approx.reshape(4, 2).astype(np.float32)
            best_area = area
    if best is None:
        return None
    s = best.sum(axis=1)
    diff = np.diff(best, axis=1)
    corners = np.float32([
        best[np.argmin(s)],
        best[np.argmin(diff)],
        best[np.argmax(s)],
        best[np.argmax(diff)],
    ])
    return corners


def _center_crop_to_aspect(img: Image.Image, aspect: float) -> Image.Image:
    w, h = img.size
    cur = w / h
    if cur > aspect:
        new_w = int(h * aspect)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    new_h = int(w / aspect)
    top = (h - new_h) // 2
    return img.crop((0, top, w, top + new_h))


def composite_type_a(scene: dict, config: dict) -> Image.Image:
    overlay = scene.get("overlay") or {}
    mockup_file = overlay.get("mockup_file") or "hand_01.png"
    screenshot_file = overlay.get("screenshot_file")

    drive_base = str(config.get("drive_base_path") or "").strip()
    base = Path(drive_base) / "assets"
    mockup_dir = base / "mockups"
    screenshots_dir = base / "screenshots"
    mockup_path = mockup_dir / mockup_file
    if not mockup_path.exists():
        pngs = sorted(mockup_dir.glob("*.png"))
        if not pngs:
            raise RuntimeError("Add phone mockup PNGs to assets/mockups/ on Drive")
        mockup_path = pngs[0]
    if not screenshot_file:
        raise RuntimeError("Type A overlay requires screenshot_file")
    screenshot_path = screenshots_dir / screenshot_file
    if not screenshot_path.exists():
        pngs = sorted(screenshots_dir.glob("*.png"))
        if not pngs:
            raise RuntimeError("Add app screenshots to assets/screenshots/ on Drive")
        screenshot_path = pngs[0]

    mockup_bgr = cv2.imread(str(mockup_path))
    if mockup_bgr is None:
        raise RuntimeError(f"Failed to read mockup: {mockup_path}")
    screenshot = Image.open(str(screenshot_path)).convert("RGB")

    corners = detect_screen_corners(mockup_bgr)
    if corners is None:
        h, w = mockup_bgr.shape[:2]
        corners = np.float32([[w * 0.30, h * 0.15], [w * 0.70, h * 0.15], [w * 0.70, h * 0.85], [w * 0.30, h * 0.85]])

    screen_w = int(np.linalg.norm(corners[1] - corners[0]))
    screen_h = int(np.linalg.norm(corners[2] - corners[1]))
    screenshot = _center_crop_to_aspect(screenshot, screen_w / screen_h).resize((screen_w, screen_h))

    src = np.float32([[0, 0], [screen_w, 0], [screen_w, screen_h], [0, screen_h]])
    M = cv2.getPerspectiveTransform(src, corners)
    warped = cv2.warpPerspective(np.array(screenshot), M, (mockup_bgr.shape[1], mockup_bgr.shape[0]))

    mask = np.zeros(mockup_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [corners.astype(np.int32)], 255)
    glow_mask = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0 * 0.12
    base_bgr = mockup_bgr.astype(np.float32)
    base_bgr = base_bgr * (1.0 - glow_mask[:, :, None]) + 255.0 * glow_mask[:, :, None]
    base_bgr = base_bgr.astype(np.uint8)

    mask_3ch = cv2.merge([mask, mask, mask])
    composited = np.where(mask_3ch > 0, warped, base_bgr)
    result = Image.fromarray(cv2.cvtColor(composited.astype(np.uint8), cv2.COLOR_BGR2RGB))
    return apply_color_grade(result, scene.get("color_grade") or "soft_warm")


def remove_white_background(image: Image.Image) -> Image.Image:
    img = image.convert("RGBA")
    data = np.array(img)
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    white = (r > 230) & (g > 230) & (b > 230)
    data[:, :, 3] = np.where(white, 0, a)
    return Image.fromarray(data)


def parse_color(color_str: str) -> tuple[int, int, int]:
    color_map = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gold": (255, 215, 0),
        "deep_purple": (48, 25, 52),
        "red": (220, 50, 50),
        "blue": (50, 100, 220),
    }
    if not color_str:
        return (255, 255, 255)
    if color_str in color_map:
        return color_map[color_str]
    if color_str.startswith("#") and len(color_str) in (4, 7):
        h = color_str.lstrip("#")
        if len(h) == 3:
            h = "".join([c * 2 for c in h])
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (255, 255, 255)


def get_position(position_str: str, base_size: tuple[int, int], element_size: tuple[int, int], padding: int = 30) -> tuple[int, int]:
    w, h = base_size
    ew, eh = element_size
    positions = {
        "top_left": (padding, padding),
        "top_center": ((w - ew) // 2, padding),
        "top_right": (w - ew - padding, padding),
        "center": ((w - ew) // 2, (h - eh) // 2),
        "bottom_left": (padding, h - eh - padding),
        "bottom_center": ((w - ew) // 2, h - eh - padding),
        "bottom_right": (w - ew - padding, h - eh - padding),
    }
    return positions.get(position_str, positions["bottom_center"])


def wrap_text(text: str, font: ImageFont.FreeTypeFont, draw: ImageDraw.ImageDraw, max_width: int) -> list[str]:
    words = (text or "").split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = current + (" " if current else "") + word
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def get_text_position(position_str: str, base_size: tuple[int, int], text_h: int, padding: int = 40) -> tuple[int, int]:
    w, h = base_size
    x = w // 2
    positions = {
        "top_center": (x, padding),
        "center": (x, (h - text_h) // 2),
        "bottom_center": (x, h - text_h - padding),
        "top_left": (padding, padding),
        "bottom_left": (padding, h - text_h - padding),
    }
    return positions.get(position_str, (x, h - text_h - padding))


def add_logo(base: Image.Image, config: dict) -> Image.Image:
    drive_base = str(config.get("drive_base_path") or "").strip()
    logo_path = Path(drive_base) / "assets" / "logo.png"
    if not logo_path.exists():
        return base
    try:
        logo = Image.open(str(logo_path)).convert("RGBA")
    except Exception:
        return base
    logo_w = int(base.width * 0.12)
    ratio = logo_w / max(1, logo.width)
    logo_h = int(logo.height * ratio)
    logo = logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
    r, g, b, a = logo.split()
    a = a.point(lambda x: int(x * 0.40))
    logo.putalpha(a)
    padding = 20
    pos = (base.width - logo_w - padding, base.height - logo_h - padding)
    base_rgba = base.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(logo, pos, logo)
    return Image.alpha_composite(base_rgba, layer).convert("RGB")


def add_symbol_overlay(base: Image.Image, overlay: dict, config: dict) -> Image.Image:
    symbol_file = overlay.get("symbol_file")
    if not symbol_file:
        return base
    niche = str(config.get("app_niche") or "generic")
    drive_base = str(config.get("drive_base_path") or "").strip()
    base_assets = Path(drive_base) / "assets" / "symbols"
    candidates = [base_assets / niche / symbol_file, base_assets / "generic" / symbol_file]
    symbol_path = next((p for p in candidates if p.exists()), None)
    if symbol_path is None:
        print(f"Symbol {symbol_file} not found, skipping", flush=True)
        return base

    symbol = remove_white_background(Image.open(str(symbol_path)).convert("RGBA"))
    symbol_w = int(base.width * 0.30)
    ratio = symbol_w / max(1, symbol.width)
    symbol_h = int(symbol.height * ratio)
    symbol = symbol.resize((symbol_w, symbol_h), Image.Resampling.LANCZOS)

    opacity = float(overlay.get("symbol_opacity", 0.65))
    opacity = max(0.0, min(1.0, opacity))
    r, g, b, a = symbol.split()
    a = a.point(lambda x: min(x, int(opacity * 255)))
    symbol.putalpha(a)

    pos = get_position(str(overlay.get("symbol_position") or "top_right"), base.size, (symbol_w, symbol_h))
    glow = symbol.filter(ImageFilter.GaussianBlur(radius=8))
    base_rgba = base.convert("RGBA")
    glow_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    glow_layer.paste(glow, pos, glow)
    base_rgba = Image.alpha_composite(base_rgba, glow_layer)
    sym_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sym_layer.paste(symbol, pos, symbol)
    base_rgba = Image.alpha_composite(base_rgba, sym_layer)
    return base_rgba.convert("RGB")


def add_text_overlay(base: Image.Image, overlay: dict, config: dict) -> Image.Image:
    text = (overlay.get("text") or "").strip()
    if not text:
        return base
    language = str(overlay.get("text_language") or "english")
    size_key = str(overlay.get("text_size") or "medium")
    color = str(overlay.get("text_color") or "white")
    position = str(overlay.get("text_position") or "bottom_center")

    font_sizes = {"large": 72, "medium": 52, "small": 36}
    base_size = int(font_sizes.get(size_key, 52) * (base.width / 1080))
    drive_base = str(config.get("drive_base_path") or "").strip()
    fonts_dir = Path(drive_base) / "assets" / "fonts"

    def _is_devanagari(s: str) -> bool:
        return any(0x0900 <= ord(c) <= 0x097F for c in s)

    if language in ("hindi", "hinglish") or _is_devanagari(text):
        font_path = fonts_dir / "NotoSansDevanagari-Bold.ttf"
    else:
        font_path = fonts_dir / "Montserrat-Bold.ttf"

    try:
        font = ImageFont.truetype(str(font_path), base_size)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            return base
        print(f"Font not found at {font_path}, using default", flush=True)

    img = base.copy()
    draw = ImageDraw.Draw(img)
    max_width = int(img.width * 0.65)
    lines = wrap_text(text, font, draw, max_width)
    line_h = base_size + 10
    block_h = len(lines) * line_h

    x, y = get_text_position(position, img.size, block_h, padding=40)
    shadow = (0, 0, 0)
    fill = parse_color(color)

    cur_y = y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        lx = x - lw // 2
        draw.text((lx + 3, cur_y + 3), line, font=font, fill=shadow)
        cur_y += line_h

    cur_y = y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        lx = x - lw // 2
        draw.text((lx, cur_y), line, font=font, fill=fill)
        cur_y += line_h

    return img


def composite_type_b(image_path: Path, scene: dict, config: dict) -> Image.Image:
    if not image_path or not Path(str(image_path)).exists():
        raise RuntimeError(f"Type B scene image not found: {image_path}")
    base = Image.open(str(image_path)).convert("RGB")
    base = apply_color_grade(base, str(scene.get("color_grade") or "warm_golden"))
    overlay = scene.get("overlay") or {}
    if overlay.get("symbol_file"):
        base = add_symbol_overlay(base, overlay, config)
    if overlay.get("text"):
        base = add_text_overlay(base, overlay, config)
    base = add_logo(base, config)
    return base


def composite_type_c(image_path: Path, scene: dict, config: dict) -> Image.Image:
    if not image_path or not Path(str(image_path)).exists():
        raise RuntimeError(f"Type C scene image not found: {image_path}")
    base = Image.open(str(image_path)).convert("RGB")
    base = apply_color_grade(base, str(scene.get("color_grade") or "soft_warm"))
    return add_logo(base, config)


def composite_scene(scene: dict, image_path: Path | None, config: dict, composited_dir: Path) -> Path:
    """
    Create final composited frame for a scene and save to Drive immediately.
    """
    scene_type = scene.get("type")
    if scene_type == "A":
        result = composite_type_a(scene, config)
    elif scene_type == "B":
        if image_path is None:
            raise RuntimeError(f"Scene {scene.get('id')} type B requires image_path")
        result = composite_type_b(image_path, scene, config)
    else:
        if image_path is None:
            raise RuntimeError(f"Scene {scene.get('id')} type C requires image_path")
        result = composite_type_c(image_path, scene, config)

    sid = scene.get("id")
    try:
        sid = int(sid) if sid is not None else 0
    except (TypeError, ValueError):
        sid = 0
    out_path = Path(composited_dir) / f"scene_{sid:02d}_composited.png"
    Path(out_path.parent).mkdir(parents=True, exist_ok=True)
    result.save(str(out_path), quality=95)
    print(f"Scene {sid} composited: {out_path}", flush=True)
    return out_path

