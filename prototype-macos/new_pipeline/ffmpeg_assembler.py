from __future__ import annotations

import os
import subprocess
from pathlib import Path

from flux_generator import get_dimensions
from utils import ensure_dir


POST_PROCESS = (
    "unsharp=5:5:0.8:3:3:0.4,"
    "noise=alls=3:allf=t+u,"
    "vignette=PI/4"
)

TEMP_VIDEO_FILENAME = "temp_video.mp4"

FFMPEG_LOGLEVEL = "error"


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    """
    Run ffmpeg with consistent logging and error reporting.
    Always injects: -hide_banner -loglevel error
    Prints stderr if returncode != 0.
    """
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", FFMPEG_LOGLEVEL] + args
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        stderr = (res.stderr or "").strip()
        if stderr:
            print(stderr[:2000], flush=True)
        raise RuntimeError(f"ffmpeg failed (exit {res.returncode})")
    return res


def _abs(p: Path) -> str:
    return str(Path(p).resolve())


def _which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def ensure_ffmpeg_installed() -> None:
    if _which("ffmpeg") and _which("ffprobe"):
        return
    # Auto-install in Colab if possible.
    try:
        if Path("/content").exists():
            subprocess.run(["apt-get", "update", "-y"], check=False, capture_output=True, text=True)
            subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=False, capture_output=True, text=True)
    except Exception:
        pass
    if _which("ffmpeg") and _which("ffprobe"):
        return
    raise RuntimeError("ffmpeg/ffprobe not found. In Colab run: apt-get install -y ffmpeg")


def get_zoompan_filter(movement: str, duration_frames: int, width: int, height: int) -> str:
    d = max(1, int(duration_frames))
    w, h = int(width), int(height)
    y_center = "ih/2-(ih/zoom/2)"
    filters = {
        "slow_zoom_in": f"zoompan=z='min(zoom+0.0008,1.3)':x='iw/2-(iw/zoom/2)':y='{y_center}':d={d}:s={w}x{h}",
        "slow_zoom_out": f"zoompan=z='max(zoom-0.0008,1.0)':x='iw/2-(iw/zoom/2)':y='{y_center}':d={d}:s={w}x{h}",
        "pan_left": f"zoompan=z='1.2':x='iw/4+on/{d}*iw/4':y='{y_center}':d={d}:s={w}x{h}",
        "pan_right": f"zoompan=z='1.2':x='iw/2-on/{d}*iw/4':y='{y_center}':d={d}:s={w}x{h}",
        "pan_top_bottom": f"zoompan=z='1.2':x='iw/2-(iw/zoom/2)':y='ih/4+on/{d}*ih/4':d={d}:s={w}x{h}",
        "diagonal_drift": f"zoompan=z='min(zoom+0.0005,1.2)':x='on/{d}*iw/8':y='on/{d}*ih/8':d={d}:s={w}x{h}",
    }
    return filters.get(movement, filters["slow_zoom_in"])


def get_ffmpeg_grade(grade: str) -> str:
    grades = {
        "warm_golden": "eq=saturation=1.3:contrast=1.1:brightness=0.02,colorchannelmixer=rr=1.1:gg=0.95:bb=0.85",
        "cool_mystical": "eq=saturation=1.2:contrast=1.15:brightness=-0.03,colorchannelmixer=rr=0.9:gg=0.95:bb=1.15",
        "dark_cinematic": "eq=saturation=0.85:contrast=1.2:brightness=-0.05,curves=r='0/0 0.5/0.4 1/0.9'",
        "soft_warm": "eq=saturation=1.1:contrast=1.05:brightness=0.05,colorchannelmixer=rr=1.05:gg=1.02:bb=0.92",
    }
    return grades.get(grade, grades["soft_warm"])


def generate_audio(scene: dict, config: dict, out_dir: Path) -> Path | None:
    from gtts import gTTS
    try:
        from langdetect import detect  # type: ignore
    except Exception:
        detect = None  # type: ignore[assignment]

    text = (scene.get("voiceover_line") or "").strip()
    if not text:
        return None
    ad_language = str(config.get("ad_language") or "english").strip().lower()
    # Simple rule: CONFIG drives language.
    if ad_language == "hindi":
        lang = "hi"
    elif ad_language == "english":
        lang = "en"
    elif ad_language == "hinglish":
        # Mixed lines: detect per line (English gTTS handles most Hinglish fine).
        if detect is not None:
            try:
                lang = "hi" if detect(text) == "hi" else "en"
            except Exception:
                lang = "en"
        else:
            lang = "en"
    else:
        lang = "en"

    duration_sec = float(scene.get("duration_sec") or 3)
    out_path = (ensure_dir(out_dir) / f"audio_{int(scene['id']):02d}.mp3").resolve()
    try:
        gTTS(text=text, lang=lang, slow=False).save(str(out_path))
        return out_path
    except Exception as e:
        print(f"gTTS failed for scene {scene.get('id')}: {e}", flush=True)
        try:
            alt = "en" if lang == "hi" else "hi"
            gTTS(text=text, lang=alt, slow=False).save(str(out_path))
            return out_path
        except Exception:
            # Silent audio fallback so pipeline continues.
            try:
                _run_ffmpeg(
                    [
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=r=44100:cl=mono",
                        "-t",
                        f"{duration_sec:.3f}",
                        _abs(out_path),
                    ]
                )
                return out_path
            except Exception as e2:
                print(f"Silent audio fallback failed for scene {scene.get('id')}: {e2}", flush=True)
                return None


def _write_concat_file(paths: list[Path], out_path: Path) -> None:
    out_path = out_path.resolve()
    out_path.write_text("".join([f"file '{str(p.resolve())}'\n" for p in paths]), encoding="utf-8")


def _merge_audio(audio_paths: list[Path], work_dir: Path) -> Path | None:
    audio_paths = [p for p in audio_paths if p and p.exists()]
    if not audio_paths:
        return None
    concat_txt = work_dir / "audio_concat.txt"
    _write_concat_file(audio_paths, concat_txt)
    merged = (work_dir / "merged_audio.mp3").resolve()
    _run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", _abs(concat_txt), "-c", "copy", _abs(merged)])
    return merged if merged.exists() else None


def _concat_video_no_transitions(clip_paths: list[Path], work_dir: Path) -> Path:
    concat_txt = work_dir / "concat.txt"
    _write_concat_file(clip_paths, concat_txt)
    out = (work_dir / TEMP_VIDEO_FILENAME).resolve()
    _run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", _abs(concat_txt), "-c", "copy", _abs(out)])
    if not out.exists():
        raise RuntimeError("FFmpeg concat failed; temp_video.mp4 not created")
    return out


def _xfade_transition_name(transition_out: str) -> str:
    t = (transition_out or "").strip().lower()
    mapping = {
        "crossfade": "fade",
        "fadeblack": "fadeblack",
        "wipeleft": "wipeleft",
    }
    return mapping.get(t, "fade")


def _assemble_with_xfade(scenes: list[dict], clip_paths: list[Path], work_dir: Path, fade_dur: float = 0.35) -> Path:
    """
    Assemble clips using xfade transitions.
    Assumes clips are in the same order as scenes (1 clip per scene).
    """
    if len(clip_paths) != len(scenes):
        # Fall back to fast concat if mismatch; better than failing hard.
        return _concat_video_no_transitions(clip_paths, work_dir)
    if not clip_paths:
        raise RuntimeError("No clips to assemble")
    if len(clip_paths) == 1:
        out = (work_dir / TEMP_VIDEO_FILENAME).resolve()
        out.write_bytes(Path(clip_paths[0]).read_bytes())
        return out

    out = (work_dir / TEMP_VIDEO_FILENAME).resolve()
    filter_parts: list[str] = []
    offsets: list[float] = []
    # offsets are cumulative clip durations minus fade overlaps
    cum = 0.0
    for i, scene in enumerate(scenes):
        dur = float(scene.get("duration_sec") or 3)
        cum += dur
        offsets.append(cum)
    # xfade offset for transition i is end time of clip i minus fade duration,
    # but with prior fades accounted for.
    # We compute sequentially using step = dur_i - fade_dur.
    offset = max(0.0, float(scenes[0].get("duration_sec") or 3) - fade_dur)
    for i in range(len(clip_paths) - 1):
        left = f"[v{i}]" if i > 0 else "[0:v]"
        right = f"[{i+1}:v]"
        out_label = f"[v{i+1}]"
        trans = _xfade_transition_name(str(scenes[i].get("transition_out") or "crossfade"))
        filter_parts.append(
            f"{left}{right}xfade=transition={trans}:duration={fade_dur:.3f}:offset={offset:.3f}{out_label}"
        )
        # next offset advances by (dur_next - fade_dur)
        dur_next = float(scenes[i + 1].get("duration_sec") or 3)
        offset += max(0.0, dur_next - fade_dur)

    filter_chain = ";".join(filter_parts)
    last_v = f"[v{len(clip_paths)-1}]"

    args: list[str] = ["-y"]
    for p in clip_paths:
        args += ["-i", _abs(Path(p))]
    args += ["-filter_complex", filter_chain, "-map", last_v, "-c:v", "libx264", "-pix_fmt", "yuv420p", _abs(out)]
    _run_ffmpeg(args)
    if not out.exists():
        raise RuntimeError("xfade assembly failed; temp_video.mp4 not created")
    return out


def _generate_clip(scene: dict, img_path: Path, fps: int, width: int, height: int, work_dir: Path) -> Path | None:
    duration_sec = float(scene.get("duration_sec") or 3)
    duration_frames = int(duration_sec * fps)
    movement = str(scene.get("camera_movement") or "slow_zoom_in")
    grade = str(scene.get("color_grade") or "soft_warm")
    zoompan = get_zoompan_filter(movement, duration_frames, width, height)
    ff_grade = get_ffmpeg_grade(grade)
    fade_out_start = max(0.1, duration_sec - 0.25)
    clip_path = (work_dir / f"clip_{int(scene['id']):02d}.mp4").resolve()
    try:
        _run_ffmpeg(
            [
                "-y",
                "-loop",
                "1",
                "-i",
                _abs(img_path),
                "-vf",
                f"{zoompan},{ff_grade},{POST_PROCESS},fade=t=in:st=0:d=0.25,fade=t=out:st={fade_out_start}:d=0.25",
                "-t",
                f"{duration_sec:.3f}",
                "-r",
                str(fps),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                _abs(clip_path),
            ]
        )
    except Exception:
        return None
    return clip_path


def _burn_subtitles(video_in: Path, srt_path: Path, video_out: Path) -> None:
    path_str = str(srt_path.resolve()).replace("'", "'\\''")
    vf = f"subtitles='{path_str}'"
    _run_ffmpeg(
        [
            "-y",
            "-i",
            _abs(video_in),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            _abs(video_out),
        ]
    )


def _ffprobe_ok(mp4_path: Path) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(mp4_path)],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0 and mp4_path.exists() and mp4_path.stat().st_size > 10_000


def assemble_video(
    scenes: list[dict],
    composited_paths: list[Path],
    audio_paths: list[Path | None],
    captions_srt: Path | None,
    config: dict,
    video_index: int,
    work_dir: Path,
    output_dir: Path,
) -> Path:
    """
    Assemble final video:
    - Generate per-scene motion clips with zoompan + grade + postprocess
    - Assemble clips with xfade transitions (crossfade/fadeblack/wipeleft)
    - Concat audio (copy) and mux
    - Optionally burn captions.srt
    - Validate with ffprobe
    """
    ensure_ffmpeg_installed()
    fps = 30
    width, height = get_dimensions(str(config.get("output_format") or "reels"))
    work_dir = ensure_dir(work_dir).resolve()
    output_dir = ensure_dir(output_dir).resolve()

    clip_paths: list[Path] = []
    clip_audio_paths: list[Path] = []
    used_scenes: list[dict] = []
    for scene, img_path, a in zip(scenes, composited_paths, audio_paths):
        if not img_path or not Path(img_path).exists():
            continue
        clip = _generate_clip(scene, Path(img_path), fps, width, height, work_dir)
        if clip:
            clip_paths.append(clip)
            used_scenes.append(scene)
            if a and Path(a).exists():
                clip_audio_paths.append(Path(a))

    if not clip_paths:
        raise RuntimeError("No clips generated. Check compositing output and FFmpeg logs.")

    merged_audio = _merge_audio(clip_audio_paths, work_dir)
    temp_video = _assemble_with_xfade(used_scenes, clip_paths, work_dir)

    out_path = (output_dir / f"video_{video_index:02d}.mp4").resolve()
    if merged_audio and merged_audio.exists():
        _run_ffmpeg(
            [
                "-y",
                "-i",
                _abs(temp_video),
                "-i",
                _abs(merged_audio),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                _abs(out_path),
            ]
        )
    else:
        out_path.write_bytes(temp_video.read_bytes())

    # Burn captions last (optional)
    if captions_srt and captions_srt.exists():
        subtitled = (work_dir / f"video_{video_index:02d}_sub.mp4").resolve()
        _burn_subtitles(out_path, captions_srt, subtitled)
        out_path.write_bytes(subtitled.read_bytes())

    if not _ffprobe_ok(out_path):
        raise RuntimeError(f"Output video validation failed for {out_path}")

    print(f"Video {video_index} saved to Drive: {out_path}", flush=True)
    return out_path

