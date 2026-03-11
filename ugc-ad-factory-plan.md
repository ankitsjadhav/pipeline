# Local UGC Ad Factory — Build Plan

## macOS adaptation (minimal local prototype)

This plan was originally written for **Windows + NVIDIA CUDA**. On macOS you’ll typically run **CPU + Apple Metal (if available)** instead. The simplest macOS prototype is:

```
[1. Script (LLM)] → [2. Voice (TTS)] → [3. Captions (Whisper)] → [4. Assembly (FFmpeg)]
    Ollama             Pocket TTS           faster-whisper          ffmpeg
```

This mirrors the Notion flow (“Script → AI Voice → Edit”) but locally:

- **Step 1 — Script**: Ollama generates **hooks + a UGC voiceover + overlay ideas + a b-roll shotlist**
- **Step 2 — AI Voice**: Pocket TTS generates `voiceover.wav` (CPU, open-source)
- **Step 3 — Edit**: use either:
  - **CapCut** (manual, closest to the Notion guide) using `output/edit_plan.md`, `voiceover.wav`, and your product clips
  - **FFmpeg** (automatic) to produce a basic `final.mp4` with burned captions

### macOS quickstart (zsh)

```bash
# 0) Install Homebrew (if needed): https://brew.sh

# 1) Core deps
brew install python@3.11 git ffmpeg ollama pocket-tts

# 2) Start Ollama (either open the Ollama app, or run:)
ollama serve

# 3) Pull a small open model (fast + good enough for prototype)
ollama pull qwen2.5:7b-instruct-q4_K_M

# 4) Create and run the minimal prototype in this repo
cd /Users/ankitjadhav/Personal/Work/ugc-ad-factory/prototype-macos
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python orchestrator.py \
  --product "Shower Filter" \
  --description "Hard water filter that reduces chlorine; improves skin and hair." \
  --audience "People with dry skin/hair who rent apartments"
```

### Windows → macOS mapping

- **Project path**: `C:\\UGC-Factory` → `~/UGC-Factory` (or keep everything in this repo)
- **Activate venv**: `\\.\\venv\\Scripts\\activate` → `source venv/bin/activate`
- **Killing Ollama**: `taskkill ... ollama.exe` → usually unnecessary on macOS; if you must: `pkill ollama`
- **CUDA steps**: skip (no NVIDIA CUDA on macOS). Prefer CPU/Metal-friendly components.

## Original Machine: NeoDesk (Windows baseline)

| Component | Spec | Impact |
|-----------|------|--------|
| CPU | i7-10750H (6C/12T, 2.6GHz) | Good for FFmpeg, Whisper, orchestration |
| RAM | 32GB DDR4 | Sufficient — can offload some model layers to RAM |
| GPU | RTX 2060 Max-Q (**6GB VRAM**) | **The bottleneck.** Rules out large models. |
| Storage | NVMe SSD | Fast model loading, good scratch disk |
| OS | Windows 11 (64-bit) | Full support via WSL2 or native |

---

## Reality Check

6GB VRAM is tight. Here's what that means:

- **Can run well:** LLMs up to 8B params (Q4), TTS models, lip-sync/talking-head models, SD 1.5 image gen, Whisper, FFmpeg compositing
- **Can run with tricks:** SDXL (with --medvram), some smaller video gen models
- **Cannot run locally:** Large video generation (Wan2.6, CogVideoX-5B, Open-Sora), FLUX image gen, 70B+ LLMs, Wan2.2-S2V full pipeline

The plan below is designed around your actual hardware. No wishful thinking.

---

## Pipeline Overview

```
[1. Script Gen]  →  [2. Voice Gen]  →  [3. Talking Head]  →  [4. B-Roll]  →  [5. Captions]  →  [6. Assembly]
   Ollama 8B         Chatterbox        SadTalker/            AnimateDiff       Whisper          FFmpeg/
   Qwen2.5 7B        or XTTS-v2        LivePortrait          or SD1.5          (local)          MoviePy
                                                              stills
```

---

## Phase 0: Foundation Setup (Do This First)

### Install Core Dependencies

```powershell
# 1. Install Python 3.10 or 3.11 (NOT 3.12+ — many AI libs break)
# Download from python.org, check "Add to PATH"

# 2. Install CUDA Toolkit 11.8
# Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive
# This matches your RTX 2060's compute capability

# 3. Install Git
# https://git-scm.com/download/win

# 4. Install FFmpeg
# Download from https://www.gyan.dev/ffmpeg/builds/
# Add to system PATH

# 5. Install Ollama (for local LLMs)
# https://ollama.com/download/windows

# 6. Create a master project folder
mkdir C:\UGC-Factory
cd C:\UGC-Factory

# 7. Create Python virtual environment
python -m venv venv
.\venv\Scripts\activate

# 8. Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
# Should show ~6.0 GB
```

---

## Phase 1: Script Generation

### What It Replaces
MakeUGC's AI script writer / Gemini prompt-based ad copywriting.

### Install

```powershell
# Pull a model that fits in 6GB VRAM comfortably
ollama pull qwen2.5:7b-instruct-q4_K_M
# Alternative: ollama pull llama3.1:8b-instruct-q4_K_M
# Alternative: ollama pull mistral:7b-instruct-q4_K_M
```

### Usage

Create a prompt template file `C:\UGC-Factory\prompts\ugc_script.txt`:

```
You are a direct-response ad copywriter specializing in UGC-style video ads.

Product: {product_name}
Description: {product_description}
Target Audience: {target_audience}
Ad Length: 30-60 seconds

Write a UGC-style video ad script with these sections:
1. HOOK (0-3s): Pattern-interrupt opening. Stop the scroll.
2. PROBLEM (3-10s): Agitate the pain point.
3. SOLUTION (10-25s): Introduce product as the answer.
4. PROOF (25-40s): Social proof, results, testimonial feel.
5. CTA (40-50s): Clear call to action with urgency.

Format each section with:
- [VISUAL]: What is shown on screen
- [VOICEOVER]: Exact script to be spoken
- [TEXT OVERLAY]: Any on-screen text

Keep the tone conversational, authentic, like a real person talking to camera.
Do NOT sound like an ad. Sound like a friend recommending something.
```

### Python Integration

```python
import requests
import json

def generate_script(product_name, description, audience):
    prompt_template = open("prompts/ugc_script.txt").read()
    prompt = prompt_template.format(
        product_name=product_name,
        product_description=description,
        target_audience=audience
    )
    
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "qwen2.5:7b-instruct-q4_K_M",
        "prompt": prompt,
        "stream": False
    })
    return json.loads(response.text)["response"]
```

### VRAM Usage: ~4.5GB (Q4 7B model)

---

## Phase 2: Voice Generation (TTS + Voice Cloning)

### What It Replaces
MakeUGC's AI voice actors.

### Option A: Chatterbox (Recommended — Best Quality)

```powershell
cd C:\UGC-Factory
pip install chatterbox-tts
```

```python
import torch
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

# Generate with emotion control
text = "Stop scrolling. I need to tell you about something that changed my morning routine."
wav = model.generate(
    text,
    audio_prompt_path="reference_voice.wav",  # 5-10s sample for cloning
    exaggeration=0.5,  # 0.0 = monotone, 1.0 = very expressive
)
model.save_wav(wav, "voiceover.wav")
```

### Option B: Coqui XTTS-v2 (Most Mature, 17 Languages)

```powershell
pip install TTS
```

```python
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
tts.tts_to_file(
    text="This product literally saved my skin routine.",
    speaker_wav="reference_voice.wav",
    language="en",
    file_path="voiceover.wav"
)
```

### Option C: OpenVoice (Fastest, Zero-Shot)

```powershell
git clone https://github.com/myshell-ai/OpenVoice.git
cd OpenVoice
pip install -e .
```

### Where to Get Reference Voice Samples

For voice cloning, you need a clean 5-15 second audio clip:
- Record yourself (best for "founder UGC" style)
- Use royalty-free voice samples from Freesound.org
- Use LibriVox audiobook samples (public domain)

### VRAM Usage: ~2-3GB (unload LLM first via `ollama stop`)

> **CRITICAL:** You cannot run Ollama + TTS simultaneously on 6GB.
> Always stop Ollama before running TTS: `ollama stop`
> The orchestrator script (Phase 7) handles this automatically.

---

## Phase 3: Talking Head / Avatar Video

### What It Replaces
MakeUGC's AI avatar actors (their core feature).

### Option A: SadTalker (Best for 6GB — Proven & Stable)

The most battle-tested talking head model that runs comfortably on 6GB.

```powershell
cd C:\UGC-Factory
git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker
pip install -r requirements.txt

# Download pretrained models (run once)
bash scripts/download_models.sh
# On Windows, manually download from their GitHub releases
```

```python
# Generate talking head video
# Input: portrait image + audio file → Output: video of person speaking
python inference.py \
    --driven_audio voiceover.wav \
    --source_image avatar.png \
    --enhancer gfpgan \
    --result_dir output/ \
    --still  # keeps head mostly still (more UGC-like)
```

### Option B: LivePortrait (Better Quality, Slightly Heavier)

```powershell
git clone https://github.com/KwaiVGI/LivePortrait.git
cd LivePortrait
pip install -r requirements.txt
```

LivePortrait uses ~4-5GB VRAM. It produces more natural head movement
and better lip sync than SadTalker but takes longer to process.

### Option C: MuseTalk (Real-Time Lip Sync)

```powershell
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk
pip install -r requirements.txt
```

MuseTalk can do real-time lip sync at ~4GB VRAM. Good for
quick iterations but less natural full-head movement.

### Where to Get Avatar Source Images

You need a front-facing portrait photo. Options:
- **Generated Photos** (generated.photos) — free AI-generated faces, royalty-free
- **This Person Does Not Exist** (thispersondoesnotexist.com) — random AI faces
- **Stable Diffusion locally** — generate custom avatars (see Phase 4)
- **Your own photo** — for founder-led UGC content

### Best Practices for Source Images
- Front-facing, neutral expression
- Good lighting, plain background
- At least 512x512 resolution
- No sunglasses or heavy occlusion

### VRAM Usage: ~3-5GB depending on model

---

## Phase 4: B-Roll & Product Visuals

### What It Replaces
MakeUGC's Sora/Veo/Kling B-roll generation.

### The Hard Truth About 6GB VRAM + Video Gen

Full text-to-video models (Wan, CogVideoX-5B, Open-Sora) need 12-24GB+ VRAM.
They will NOT run on your 2060 Max-Q. Here are realistic alternatives:

### Strategy A: AI Image Generation → Ken Burns Animation (Recommended)

Generate beautiful product/lifestyle stills, then animate them into B-roll
using pan/zoom/parallax effects. This is what 90% of successful UGC ads
actually use anyway — they're image-based, not video-based.

```powershell
# Install Stable Diffusion WebUI (A1111)
cd C:\UGC-Factory
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# Download SD 1.5 model (fits in 6GB perfectly)
# Place .safetensors file in models/Stable-diffusion/

# Launch with optimizations for 6GB
python launch.py --medvram --xformers
```

Generate product-in-context images:
- Product on kitchen counter, lifestyle setting
- Before/after comparison shots
- Close-up product detail shots
- Person using the product (img2img with your avatar)

Then animate with FFmpeg Ken Burns effect:

```bash
# Slow zoom in (5 seconds, 30fps)
ffmpeg -loop 1 -i product_shot.png -vf "zoompan=z='min(zoom+0.001,1.3)':d=150:s=1080x1920:fps=30" -t 5 -c:v libx264 broll_zoom.mp4

# Slow pan left to right (5 seconds)
ffmpeg -loop 1 -i lifestyle.png -vf "zoompan=z='1.2':x='iw/2-(iw/zoom/2)+((iw/zoom)*on/150)':d=150:s=1080x1920:fps=30" -t 5 -c:v libx264 broll_pan.mp4
```

### Strategy B: AnimateDiff (Actual Video Gen on 6GB)

AnimateDiff with SD 1.5 can generate short 2-3 second video clips
and runs within 6GB VRAM.

```powershell
# Install via ComfyUI (better VRAM management than A1111)
cd C:\UGC-Factory
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

# Install AnimateDiff node
cd custom_nodes
git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git
```

Use AnimateDiff for:
- Subtle product rotation/movement
- Background motion (clouds, water, particles)
- Lifestyle scene animations

### Strategy C: Stock Footage (Zero Compute)

Don't overlook free stock video:
- **Pexels** (pexels.com) — completely free, no attribution needed
- **Pixabay** (pixabay.com) — free stock video
- **Coverr** (coverr.io) — free B-roll clips

Many top UGC ads use stock B-roll anyway. The talking head is
what makes it feel "UGC" — the B-roll just supports the narrative.

### VRAM Usage: ~4-6GB for SD 1.5 / AnimateDiff

---

## Phase 5: Captions & Subtitles

### What It Replaces
MakeUGC's auto-captioning.

### Install Whisper Locally

```powershell
pip install openai-whisper
# Or for faster inference:
pip install faster-whisper
```

### Generate Word-Level Timestamps

```python
from faster_whisper import WhisperModel

# "small" model runs great on CPU with 32GB RAM
# "medium" also works but slower
model = WhisperModel("small", device="cpu", compute_type="int8")

segments, info = model.transcribe("voiceover.wav", word_timestamps=True)

# Generate SRT file
srt_content = ""
idx = 1
for segment in segments:
    for word in segment.words:
        start = format_timestamp(word.start)
        end = format_timestamp(word.end)
        srt_content += f"{idx}\n{start} --> {end}\n{word.word.strip()}\n\n"
        idx += 1

with open("captions.srt", "w") as f:
    f.write(srt_content)

def format_timestamp(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"
```

### Burn Captions Into Video (UGC Style)

```bash
# Bold, centered, word-by-word captions (TikTok/Reels style)
ffmpeg -i talking_head.mp4 -vf "subtitles=captions.srt:force_style='FontName=Arial Black,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=3,Alignment=2,MarginV=80'" -c:a copy captioned.mp4
```

### VRAM Usage: 0GB (runs on CPU) or ~1.5GB on GPU

---

## Phase 6: Final Video Assembly

### What It Replaces
MakeUGC's ad maker / auto-composer.

### Assembly Script (Python + FFmpeg)

```python
from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip,
    concatenate_videoclips, CompositeVideoClip,
    TextClip, vfx
)

def assemble_ugc_ad(
    talking_head_path,
    broll_paths,
    voiceover_path,
    output_path="final_ad.mp4",
    resolution=(1080, 1920)  # 9:16 vertical
):
    # Load talking head
    talking = VideoFileClip(talking_head_path)
    
    # Load B-roll clips
    brolls = [VideoFileClip(p) for p in broll_paths]
    
    # Build timeline:
    # Hook (talking) → B-roll 1 → Talking → B-roll 2 → CTA (talking)
    timeline = [
        talking.subclip(0, 5),                    # Hook
        brolls[0].resize(resolution).subclip(0,3), # Product shot
        talking.subclip(5, 20),                    # Body
        brolls[1].resize(resolution).subclip(0,3) if len(brolls) > 1 else None,
        talking.subclip(20, None),                 # CTA
    ]
    timeline = [c for c in timeline if c is not None]
    
    # Concatenate with crossfade
    final = concatenate_videoclips(timeline, method="compose")
    
    # Add background music (low volume)
    # Get royalty-free music from pixabay.com/music
    # bgm = AudioFileClip("bgm.mp3").volumex(0.1)
    # final = final.set_audio(CompositeAudioClip([final.audio, bgm]))
    
    final.write_videofile(
        output_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="medium"
    )

# Install: pip install moviepy
```

### Add CTA Overlay with FFmpeg

```bash
# Add "Shop Now" text overlay in last 5 seconds
ffmpeg -i final_ad.mp4 -vf "drawtext=text='Shop Now 👇':fontfile=Arial:fontsize=48:fontcolor=white:borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-200:enable='gte(t,25)'" -c:a copy final_with_cta.mp4
```

---

## Phase 7: The Orchestrator (Ties Everything Together)

Save as `C:\UGC-Factory\orchestrator.py`:

```python
"""
Local UGC Ad Factory Orchestrator
Runs the full pipeline sequentially, managing VRAM by loading/unloading models.

Usage: python orchestrator.py --product "Product Name" --description "..." --audience "..."
"""

import subprocess
import os
import argparse
import json
import time

PROJECT_DIR = r"C:\UGC-Factory"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

def step(name):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}\n")

def stop_ollama():
    """Free VRAM by stopping Ollama"""
    subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], 
                   capture_output=True)
    time.sleep(2)

def start_ollama():
    """Start Ollama server"""
    subprocess.Popen(["ollama", "serve"], 
                     stdout=subprocess.DEVNULL, 
                     stderr=subprocess.DEVNULL)
    time.sleep(5)

def generate_script(product, description, audience):
    step("1/6 — Generating Ad Script")
    start_ollama()
    
    import requests
    prompt = f"""Write a 45-second UGC video ad script for:
    Product: {product}
    Description: {description}
    Target: {audience}
    
    Format with [HOOK], [PROBLEM], [SOLUTION], [PROOF], [CTA] sections.
    Include [VOICEOVER] text for each section.
    Keep it conversational, authentic, like a real person."""
    
    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": "qwen2.5:7b-instruct-q4_K_M",
        "prompt": prompt,
        "stream": False
    })
    script = json.loads(resp.text)["response"]
    
    script_path = os.path.join(OUTPUT_DIR, "script.txt")
    with open(script_path, "w") as f:
        f.write(script)
    
    # Extract just the voiceover text for TTS
    # (You may need to parse this more carefully)
    voiceover_text = extract_voiceover(script)
    vo_path = os.path.join(OUTPUT_DIR, "voiceover_text.txt")
    with open(vo_path, "w") as f:
        f.write(voiceover_text)
    
    stop_ollama()  # Free VRAM for next step
    return script_path, vo_path

def extract_voiceover(script):
    """Extract [VOICEOVER] lines from script"""
    lines = []
    for line in script.split("\n"):
        if "VOICEOVER" in line.upper():
            text = line.split(":", 1)[-1].strip().strip('"')
            lines.append(text)
    return " ".join(lines) if lines else script

def generate_voice(text_path, reference_wav=None):
    step("2/6 — Generating Voiceover")
    text = open(text_path).read()
    
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    
    output_path = os.path.join(OUTPUT_DIR, "voiceover.wav")
    
    if reference_wav:
        tts.tts_to_file(text=text, speaker_wav=reference_wav,
                        language="en", file_path=output_path)
    else:
        tts.tts_to_file(text=text, language="en", file_path=output_path)
    
    del tts  # Free VRAM
    import torch; torch.cuda.empty_cache()
    return output_path

def generate_talking_head(avatar_path, audio_path):
    step("3/6 — Generating Talking Head Video")
    output_path = os.path.join(OUTPUT_DIR, "talking_head.mp4")
    
    # Using SadTalker (most reliable on 6GB)
    subprocess.run([
        "python", os.path.join(PROJECT_DIR, "SadTalker", "inference.py"),
        "--driven_audio", audio_path,
        "--source_image", avatar_path,
        "--enhancer", "gfpgan",
        "--result_dir", OUTPUT_DIR,
        "--still",
        "--preprocess", "crop"
    ])
    
    import torch; torch.cuda.empty_cache()
    return output_path

def generate_captions(audio_path):
    step("4/6 — Generating Captions")
    from faster_whisper import WhisperModel
    
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)
    
    srt_path = os.path.join(OUTPUT_DIR, "captions.srt")
    with open(srt_path, "w") as f:
        idx = 1
        for seg in segments:
            for word in seg.words:
                start = format_ts(word.start)
                end = format_ts(word.end)
                f.write(f"{idx}\n{start} --> {end}\n{word.word.strip()}\n\n")
                idx += 1
    
    del model
    return srt_path

def generate_broll(product_name):
    step("5/6 — B-Roll (Manual Step)")
    print(f"Generate product images using Stable Diffusion WebUI")
    print(f"Suggested prompts:")
    print(f'  - "product photo of {product_name}, lifestyle, bright, clean"')
    print(f'  - "person using {product_name}, candid, natural light"')
    print(f"Or use stock footage from pexels.com")
    print(f"\nPlace B-roll files in: {OUTPUT_DIR}\\broll\\")
    input("Press Enter when B-roll is ready...")
    
    broll_dir = os.path.join(OUTPUT_DIR, "broll")
    broll_files = [os.path.join(broll_dir, f) 
                   for f in os.listdir(broll_dir) 
                   if f.endswith(('.mp4', '.mov', '.png', '.jpg'))]
    return broll_files

def assemble_final(talking_head, captions_srt, broll_files):
    step("6/6 — Assembling Final Ad")
    output = os.path.join(OUTPUT_DIR, "final_ugc_ad.mp4")
    
    # Burn captions into talking head
    captioned = os.path.join(OUTPUT_DIR, "captioned.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", talking_head,
        "-vf", f"subtitles={captions_srt}:force_style='FontName=Arial Black,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=3,Alignment=2,MarginV=100'",
        "-c:a", "copy",
        captioned
    ])
    
    print(f"\n{'='*60}")
    print(f"  DONE! Final ad: {captioned}")
    print(f"{'='*60}")
    return captioned

def format_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--audience", required=True)
    parser.add_argument("--avatar", default="avatar.png")
    parser.add_argument("--voice-ref", default=None, 
                        help="Reference voice WAV for cloning")
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "broll"), exist_ok=True)
    
    # Run pipeline (sequential — one model at a time to fit in 6GB)
    script_path, vo_text = generate_script(
        args.product, args.description, args.audience)
    
    audio_path = generate_voice(vo_text, args.voice_ref)
    
    talking_head = generate_talking_head(args.avatar, audio_path)
    
    captions = generate_captions(audio_path)
    
    broll = generate_broll(args.product)
    
    final = assemble_final(talking_head, captions, broll)
```

---

## VRAM Management Strategy (Critical for 6GB)

The #1 rule: **never run two AI models simultaneously.**

The orchestrator handles this by running steps sequentially:

| Step | Model Loaded | VRAM Used | Notes |
|------|-------------|-----------|-------|
| 1. Script | Qwen 7B Q4 via Ollama | ~4.5GB | Ollama killed after |
| 2. Voice | XTTS-v2 or Chatterbox | ~2.5GB | Model deleted after |
| 3. Talking Head | SadTalker | ~3.5GB | Model deleted after |
| 4. Captions | Whisper small | 0GB | Runs on CPU |
| 5. B-Roll | SD 1.5 (separate process) | ~4GB | Manual step |
| 6. Assembly | FFmpeg | 0GB | CPU only |

Always call `torch.cuda.empty_cache()` between steps.

---

## Folder Structure

```
C:\UGC-Factory\
├── orchestrator.py          # Main pipeline script
├── prompts\
│   └── ugc_script.txt       # Script generation prompt template
├── assets\
│   ├── avatars\             # Portrait images for talking heads
│   ├── voices\              # Reference voice samples for cloning
│   └── music\               # Background music tracks (royalty-free)
├── SadTalker\               # Cloned repo
├── stable-diffusion-webui\  # Cloned repo (for B-roll images)
├── output\                  # Generated files per run
│   ├── script.txt
│   ├── voiceover.wav
│   ├── talking_head.mp4
│   ├── captions.srt
│   ├── broll\
│   └── final_ugc_ad.mp4
└── venv\                    # Python virtual environment
```

---

## Install Checklist (Do In Order)

- [ ] Python 3.10/3.11 installed
- [ ] CUDA Toolkit 11.8 installed
- [ ] Git installed
- [ ] FFmpeg installed and in PATH
- [ ] Ollama installed + model pulled (`qwen2.5:7b-instruct-q4_K_M`)
- [ ] Virtual environment created and activated
- [ ] PyTorch with CUDA installed
- [ ] `pip install TTS faster-whisper moviepy`
- [ ] SadTalker cloned + models downloaded
- [ ] Stable Diffusion WebUI cloned + SD 1.5 model downloaded
- [ ] Test avatar image placed in `assets/avatars/`
- [ ] Test run of `orchestrator.py` end-to-end

---

## Quality Upgrade Path

When you're ready to invest in better hardware:

| Upgrade | Cost (USD) | What It Unlocks |
|---------|-----------|-----------------|
| RTX 4060 Ti 16GB (eGPU) | ~$400 | SDXL, larger TTS, better talking heads |
| RTX 4090 24GB (desktop) | ~$1,600 | Full video gen (Wan2.6, CogVideoX), FLUX, 70B LLMs |
| Cloud GPU (RunPod/Vast) | $0.50-2/hr | Run anything on-demand, no hardware purchase |

### Hybrid Approach (Best Value Right Now)

Run script gen, TTS, captions, and assembly locally (free).
Use **RunPod** or **Vast.ai** for the heavy GPU tasks:
- Rent an A100 80GB for $1.50/hr
- Run Wan2.2-S2V for talking heads
- Run Wan2.6 for B-roll video gen
- Total cost per ad: ~$0.50-1.00

---

## Comparison: Your Local Setup vs MakeUGC

| Feature | MakeUGC ($49-249/mo) | Your Local Setup ($0) |
|---------|---------------------|----------------------|
| Script writing | Gemini/GPT | Qwen 7B (local) |
| Voice quality | Professional | Good (XTTS-v2) |
| Talking head quality | OmniHuman/Nova V2 | SadTalker (decent) |
| B-roll video gen | Sora/Veo/Kling | SD 1.5 stills + Ken Burns |
| Speed per ad | 2-10 min | 15-30 min |
| Monthly cost | $49-249 | $0 (electricity only) |
| Privacy | Cloud-based | 100% local |
| Customization | Limited templates | Unlimited |
| Quality ceiling | Higher | Lower (on 6GB) |

### Honest Assessment

Your local setup will produce **B+ quality** UGC ads on 6GB VRAM.
The talking heads won't match MakeUGC's OmniHuman quality, and you
won't get AI-generated video B-roll. But for testing ad concepts,
iterating on scripts, and creating MVP ads for Sourced or your
other ventures, this pipeline is more than sufficient — and it
costs nothing to run.

The winning strategy: use this local pipeline for rapid iteration
and A/B testing concepts, then invest in a MakeUGC subscription
(or cloud GPU time) only for the final "hero" versions of your
best-performing ad concepts.
