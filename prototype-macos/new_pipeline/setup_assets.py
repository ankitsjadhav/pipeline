from __future__ import annotations

import os
from pathlib import Path

import requests

from utils import create_folder_structure, ensure_dir


FONTS: dict[str, str] = {
    "NotoSansDevanagari-Bold.ttf": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Bold.ttf",
    "Montserrat-Bold.ttf": "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf",
    "NotoSans-Regular.ttf": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
}


def _download(url: str, out_path: Path, timeout_s: int = 30) -> None:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    out_path.write_bytes(r.content)


def setup_assets(config: dict) -> dict[str, Path]:
    """
    Run once in Colab before the first pipeline run:
    - Creates the Drive folder structure
    - Downloads required fonts to assets/fonts
    """
    paths = create_folder_structure(config)
    fonts_dir = ensure_dir(paths["assets"] / "fonts")

    for filename, url in FONTS.items():
        out_path = fonts_dir / filename
        try:
            if out_path.exists() and out_path.stat().st_size > 10_000:
                print(f"Font exists: {filename}", flush=True)
                continue
            print(f"Downloading font: {filename}", flush=True)
            _download(url, out_path)
            print(f"Saved: {out_path}", flush=True)
        except Exception as e:
            raise RuntimeError(f"Failed to download font {filename} from {url}: {e}") from e

    # Ensure niche symbol folders exist (user populates them)
    base = paths["assets"] / "symbols"
    for folder in ("astrology", "fitness", "finance", "food", "generic"):
        ensure_dir(base / folder)

    # Placeholder logo path
    logo_path = paths["assets"] / "logo.png"
    if not logo_path.exists():
        print(f"NOTE: Add your transparent logo at: {logo_path}", flush=True)

    print("\n=== SETUP COMPLETE ===", flush=True)
    print("Now manually add these to Google Drive:", flush=True)
    print("  1) Phone mockup PNGs → assets/mockups/", flush=True)
    print("  2) App screenshot PNGs → assets/screenshots/", flush=True)
    print("  3) Brand logo PNG → assets/logo.png", flush=True)
    print("  4) Niche symbol PNGs → assets/symbols/<your_niche>/", flush=True)

    return paths

