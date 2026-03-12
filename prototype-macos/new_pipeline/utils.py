from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except Exception:
        return False


def mount_drive() -> None:
    if not is_colab():
        raise RuntimeError("Google Colab not detected. Run this pipeline inside Colab.")
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)


def ensure_dir(p: str | Path) -> Path:
    path = Path(str(p))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _glob_images(folder: Path) -> list[Path]:
    """Return paths for *.png and *.jpg in folder (mockups, screenshots, symbols, etc.)."""
    return list(folder.glob("*.png")) + list(folder.glob("*.jpg"))


def create_folder_structure(config: dict) -> dict[str, Path]:
    drive_base = config.get("drive_base_path")
    if not drive_base or not str(drive_base).strip():
        raise RuntimeError("CONFIG['drive_base_path'] is missing or empty. Set it to your Google Drive path (e.g. /content/drive/MyDrive/ugc_pipeline).")
    base = Path(str(drive_base).strip())
    # MyDrive/hf_cache is outside base per spec
    mydrive = Path("/content/drive/MyDrive")
    hf_cache = ensure_dir(mydrive / "hf_cache")

    assets = ensure_dir(base / "assets")
    ensure_dir(assets / "mockups")
    ensure_dir(assets / "screenshots")
    ensure_dir(assets / "fonts")
    ensure_dir(assets / "symbols" / "generic")
    ensure_dir(assets / "symbols" / "astrology")
    ensure_dir(assets / "symbols" / "fitness")
    ensure_dir(assets / "symbols" / "finance")
    ensure_dir(assets / "symbols" / "food")

    output = ensure_dir(base / "output")
    images = ensure_dir(base / "images")
    composited = ensure_dir(base / "composited")

    return {
        "mydrive": mydrive,
        "hf_cache": hf_cache,
        "base": base,
        "assets": assets,
        "output": output,
        "images": images,
        "composited": composited,
    }


def check_drive_space(min_free_gb: float = 1.0) -> None:
    drive_path = Path("/content/drive")
    if not drive_path.exists():
        raise RuntimeError("Google Drive not mounted at /content/drive. Run mount_drive() first (e.g. from google.colab import drive; drive.mount('/content/drive')).")
    total, used, free = shutil.disk_usage("/content/drive")
    free_gb = free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Drive storage critically low: {free_gb:.1f}GB free. Need at least {min_free_gb:.1f}GB."
        )
    if free_gb < 5:
        print(f"WARNING: Drive storage low: {free_gb:.1f}GB free", flush=True)


def check_required_assets(paths: dict[str, Path]) -> None:
    assets = Path(str(paths["assets"]))
    mockups = assets / "mockups"
    screenshots = assets / "screenshots"
    if not mockups.exists() or not any(_glob_images(mockups)):
        raise RuntimeError("No mockup images (PNG/JPG) found in assets/mockups/. Add at least 1 image and rerun.")
    if not screenshots.exists() or not any(_glob_images(screenshots)):
        raise RuntimeError("No app screenshot images (PNG/JPG) found in assets/screenshots/. Add at least 1 image and rerun.")


def list_png_files(folder: Path | str) -> list[str]:
    folder_p = Path(str(folder))
    if not folder_p.exists():
        return []
    return sorted([p.name for p in _glob_images(folder_p)])


def list_symbol_files(base_assets: Path | str, niche: str) -> list[str]:
    files: list[str] = []
    base_p = Path(str(base_assets))
    niche_dir = base_p / "symbols" / niche
    generic_dir = base_p / "symbols" / "generic"
    for d in (niche_dir, generic_dir):
        if d.exists():
            files.extend([p.name for p in _glob_images(d)])
    return sorted(set(files))


# Checklist compatibility wrappers
def get_available_files(config: dict, subfolder: str) -> list[str]:
    drive_base = str(config.get("drive_base_path") or "").strip()
    base = Path(drive_base) / "assets" / subfolder
    return list_png_files(base)


def get_available_symbols(config: dict) -> list[str]:
    drive_base = str(config.get("drive_base_path") or "").strip()
    assets = Path(drive_base) / "assets"
    niche = str(config.get("app_niche") or "generic")
    return list_symbol_files(assets, niche)


# Text/layout helpers (used by compositor; duplicated here to satisfy checklist)
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


def get_position(
    position_str: str,
    base_size: tuple[int, int],
    element_size: tuple[int, int],
    padding: int = 30,
) -> tuple[int, int]:
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


def get_text_position(
    position_str: str,
    base_size: tuple[int, int],
    text_h: int,
    padding: int = 40,
) -> tuple[int, int]:
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


def wrap_text(text: str, font, draw, max_width: int) -> list[str]:
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


def run(cmd: list[str], *, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def retry_sleep(attempt: int, base_s: float = 2.0, max_s: float = 10.0) -> None:
    time.sleep(min(max_s, base_s * (attempt + 1)))


def safe_json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

