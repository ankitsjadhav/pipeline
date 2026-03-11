# UGC Ad Factory

**Main:** Colab (`colab/`) — full pipeline in cloud, Ollama for script. **Fallback:** local Mac (`orchestrator.py`).

This is a **minimal local** “UGC ad factory” prototype:

- **LLM**: Ollama (local) generating a voiceover script
- **TTS**: Pocket TTS (best) or macOS `say` fallback generating `voiceover.wav`
- **Captions**: faster-whisper generating `captions.srt`
- **B-roll (Phase 4)**: **Pexels API** fetches one portrait image per shot using the shot description as search query; if `PEXELS_API_KEY` is not set or a fetch fails, text placeholders are used. Images are saved as `output/broll/scene_XXX.png`, turned into short clips (Ken Burns), and assembled with voice + captions.
- **Assembly**: FFmpeg producing a vertical MP4 (B-roll + voiceover + captions)

## Steps (do these first)

### 1) Install dependencies

```bash
# Homebrew (if needed): https://brew.sh
brew install python@3.11 ffmpeg ollama
# Optional — better voice quality (otherwise macOS `say` is used):
brew install pocket-tts
```

### 2) Start Ollama and pull a model

```bash
ollama serve
ollama pull qwen2.5:7b-instruct-q4_K_M
```

### 3) Python env + deps

```bash
cd /Users/ankitjadhav/Personal/Work/ugc-ad-factory/prototype-macos
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4) Run the prototype

```bash
python orchestrator.py \
  --product "Shower Filter" \
  --description "Hard water filter that reduces chlorine; improves skin and hair." \
  --audience "People with dry skin/hair who rent apartments"
```

Outputs land in `prototype-macos/output/`:

- `brief.json`
- `ad_assets.json` (hooks, overlays, shotlist, CTA, voiceover)
- `hooks.txt`
- `edit_plan.md` (CapCut-friendly checklist)
- `voiceover.txt`
- `voiceover.wav`
- `captions.srt`
- `final.mp4`

## Main pipeline: Google Colab

To run the **full pipeline in the cloud** (recommended): use the Colab notebook in **`colab/`**. It uses **Ollama + Qwen 2.5 7B** (same model as local) for script — no API key, no rate limits. Heavy steps (image processing, clip generation, animations, FFmpeg) run in Colab for better performance. See `colab/README.md`.

## Useful flags

- `--ollama-model`: change the local LLM (default `qwen2.5:7b-instruct-q4_K_M`)
- `--tts-voice`: Pocket TTS voice (default is a Hugging Face voice URL)
- `--whisper-model`: `tiny|base|small|medium|large-v3` (default `small`)

## Phase 4 B-roll (Pexels API)

B-roll images are **fetched from Pexels** using each `broll_shotlist` entry as the search query. One **portrait** photo per shot is downloaded, resized/cropped to 1080×1920, and saved as `output/broll/scene_000.png`, etc. The existing Ken Burns → FFmpeg pipeline then builds the final video with real visuals + voiceover + captions.

- **API key**: Get a free key at [pexels.com/api](https://www.pexels.com/api/) and set `export PEXELS_API_KEY=your_key` before running.
- **Fallback**: If the key is missing or a search returns no result, that shot uses a text-on-color placeholder.

