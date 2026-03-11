from __future__ import annotations

import shutil
import time
from pathlib import Path

from .ffmpeg_assembler import assemble_video, generate_audio
from .flux_generator import generate_image, load_flux_model
from .groq_director import generate_scene_plan
from .smart_compositor import composite_scene
from .utils import (
    check_drive_space,
    check_required_assets,
    create_folder_structure,
    ensure_dir,
    mount_drive,
)


# User edits this dict in Colab before running.
CONFIG = {
    "app_name": "YourAppName",
    "app_niche": "astrology",
    "app_description": "One or two sentence description of what the app does",
    "target_audience": "Who the ad is for, e.g. young Indian women 18-30",
    "ad_language": "hindi",
    "video_count": 8,
    "output_format": "reels",  # reels | square | landscape
    "groq_api_key": "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",
    "drive_base_path": "/content/drive/MyDrive/ugc_pipeline",
}


def _cleanup_temp(paths: dict[str, Path]) -> None:
    for folder in ("images", "composited"):
        p = paths.get(folder)
        if p and p.exists():
            try:
                shutil.rmtree(p)
            except Exception:
                pass
            p.mkdir(parents=True, exist_ok=True)
    print("Temp files cleared", flush=True)


def _generate_captions_whisper(audio_path: Path, srt_out: Path, model_name: str = "base") -> None:
    """
    Keeps faster-whisper in the pipeline as required.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError(f"faster-whisper not installed: {e}") from e

    srt_out.parent.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(audio_path), word_timestamps=False, vad_filter=True)

    def _fmt(ts: float) -> str:
        ms = int(round(ts * 1000))
        s = ms // 1000
        ms = ms % 1000
        m = s // 60
        s = s % 60
        h = m // 60
        m = m % 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: list[str] = []
    idx = 1
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        lines.append(str(idx))
        lines.append(f"{_fmt(float(seg.start))} --> {_fmt(float(seg.end))}")
        lines.append(text)
        lines.append("")
        idx += 1

    if not lines:
        raise RuntimeError("No captions generated. Check audio and Whisper output.")
    srt_out.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(config: dict | None = None) -> None:
    cfg = dict(CONFIG)
    if config:
        cfg.update(config)

    if not str(cfg.get("groq_api_key") or "").strip() or str(cfg.get("groq_api_key")).strip() == "your_groq_api_key_here":
        raise RuntimeError("Missing CONFIG['groq_api_key']. Add your Groq API key and rerun.")

    mount_drive()
    paths = create_folder_structure(cfg)
    check_drive_space(min_free_gb=1.0)
    check_required_assets(paths)

    # Load FLUX once for the whole session.
    pipe = load_flux_model(cfg)

    completed = 0
    errors: list[str] = []

    for video_idx in range(1, int(cfg.get("video_count") or 8) + 1):
        output_path = paths["output"] / f"video_{video_idx:02d}.mp4"
        if output_path.exists() and output_path.stat().st_size > 100_000:
            print(f"Video {video_idx} already exists, skipping", flush=True)
            completed += 1
            continue

        print(f"\n=== Generating Video {video_idx}/{cfg['video_count']} ===", flush=True)
        try:
            plan = generate_scene_plan(cfg, assets_base_path=paths["assets"])
            scenes = plan["scenes"]

            composited_paths: list[Path] = []
            audio_paths: list[Path | None] = []

            # Per-video working directory (Drive-backed)
            work_dir = ensure_dir(paths["base"] / "images" / f"work_{video_idx:02d}")

            for scene in scenes:
                # 1) FLUX background (Type A returns None)
                img_path = generate_image(pipe, scene, cfg, images_dir=paths["images"])

                # 2) Composite overlays (always saves to Drive)
                comp = composite_scene(scene, img_path, cfg, composited_dir=paths["composited"])
                composited_paths.append(comp)

                # 3) Per-scene voiceover line audio
                a = generate_audio(scene, cfg, out_dir=paths["images"])
                audio_paths.append(a)

            # 4) Captions for merged audio (after assembly audio concat)
            # We generate captions after audio concat by assembling once without captions, then running Whisper on merged audio.
            # To keep modules independent, we generate a merged audio file via assembler's internal concat step by reusing audio_paths.
            merged_audio = work_dir / "merged_audio.mp3"
            # Create merged audio by concat (ffmpeg) even before final mux.
            # This mirrors the assembler logic; duplicated here to keep captions generation explicit.
            audio_txt = work_dir / "audio_concat.txt"
            existing_audio = [p for p in audio_paths if p and Path(p).exists()]
            if existing_audio:
                audio_txt.write_text("".join([f"file '{str(Path(p).resolve())}'\n" for p in existing_audio]), encoding="utf-8")
                import subprocess

                subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(audio_txt), "-c", "copy", str(merged_audio)],
                    capture_output=True,
                )
            captions_srt = work_dir / "captions.srt"
            if merged_audio.exists() and merged_audio.stat().st_size > 10_000:
                _generate_captions_whisper(merged_audio, captions_srt, model_name="base")
            else:
                captions_srt = None  # no captions if no audio

            # 5) Assemble final video (saves to Drive)
            assemble_video(
                scenes=scenes,
                composited_paths=composited_paths,
                audio_paths=audio_paths,
                captions_srt=captions_srt,
                config=cfg,
                video_index=video_idx,
                work_dir=work_dir,
                output_dir=paths["output"],
            )

            completed += 1
            print(f"Video {video_idx} complete", flush=True)

        except Exception as e:
            msg = f"Video {video_idx}: {e}"
            print(f"Video {video_idx} failed: {e}", flush=True)
            errors.append(msg)

        time.sleep(5)

    _cleanup_temp(paths)
    print("\n=== PIPELINE COMPLETE ===", flush=True)
    print(f"Completed: {completed}/{cfg['video_count']}", flush=True)
    if errors:
        print(f"Errors: {len(errors)}", flush=True)
        for e in errors:
            print(f"  - {e}", flush=True)


if __name__ == "__main__":
    run_pipeline()

