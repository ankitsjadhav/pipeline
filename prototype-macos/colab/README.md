# UGC Ad Factory — Google Colab (main pipeline)

Run the **entire pipeline in Colab** (no local Mac needed): script (Ollama + Qwen 2.5 7B, same as local), voice (gTTS), captions (faster-whisper), Pexels images, clip generation, animations, and FFmpeg render. No external LLM API — uses Ollama in Colab to avoid rate limits. Heavy steps run in Colab for better performance.

## Quick start

1. Open **`ugc_ad_factory_colab.ipynb`** in Google Colab (upload the file or open from Drive).
2. Run the **Install Ollama** cell first (installs Ollama + pulls qwen2.5:7b-instruct-q4_K_M).
3. Get a **Pexels API key** (optional): https://www.pexels.com/api/ — for real B-roll images; otherwise placeholders.
4. Upload **`colab_pipeline.py`** into the Colab session (left sidebar → Files → Upload), or clone the repo and `%cd` into `prototype-macos/colab`.
5. Run all cells. Download `final.mp4` from the last cell.

## What runs in Colab

| Step | Local (Mac) | Colab |
|------|-------------|--------|
| Script | Ollama | **Ollama** (same model) |
| Voice | Pocket TTS / say | **gTTS** |
| Captions | faster-whisper | faster-whisper (CPU, base model) |
| B-roll images | Pexels | Pexels (same; improved queries) |
| Clips + assembly | FFmpeg | FFmpeg (apt install) |

## Optimizations

- **Memory**: Whisper uses CPU + int8; model is unloaded after captions. Default Whisper model is `base` (smaller than local `small`).
- **Speed**: Ollama (same model as local) for script; gTTS is quick; FFmpeg runs natively in the VM; single-pass render and optional GPU encoding when available.
- **Visual accuracy**: Prompt asks for “clear, visual search phrase” per shot; Pexels query builder prioritizes product keywords and visual terms (person, smartphone, portrait, etc.) for better stock photo matches.

## Files

- **`ugc_ad_factory_colab.ipynb`** — Notebook to open in Colab.
- **`colab_pipeline.py`** — Pipeline logic (script, TTS, captions, Pexels, Ken Burns, assembly). Must be in the Colab runtime (upload or clone).
- **`requirements-colab.txt`** — Optional: `pip install -r requirements-colab.txt` instead of the notebook’s pip cell.

The **local pipeline** in `../orchestrator.py` is unchanged and still runs on your Mac.
