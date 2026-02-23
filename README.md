# 🎙️ Empathy Engine — Emotion-Aware Neural Text-to-Speech

> An AI-powered TTS service that detects the emotion in any input text and dynamically
> modulates vocal characteristics — pitch, speed, and expressiveness — to produce
> emotionally resonant, human-like speech output.

---

## 📌 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Emotion-to-Voice Mapping](#emotion-to-voice-mapping)
- [Intensity Scaling](#intensity-scaling)
- [Code Structure](#code-structure)
- [Tech Stack](#tech-stack)
- [Setup and Installation](#setup-and-installation)
- [Running the App](#running-the-app)
- [Web Interface](#web-interface)
- [Example Output](#example-output)
- [Design Choices](#design-choices)

---

## Overview

Standard Text-to-Speech systems are functional but emotionally flat — every sentence
sounds the same regardless of whether the speaker is excited, frustrated, or calm.
The **Empathy Engine** bridges that gap.

It takes a sentence as input, classifies it into one of **7 emotional states**, and
synthesizes speech where the vocal parameters — speaking rate, noise expressiveness,
and pitch — are all tuned specifically to that emotion. The intensity of those
modulations is further scaled by the model's confidence score, so a mildly positive
sentence sounds subtly upbeat while an ecstatically positive one sounds genuinely
enthusiastic.

---

## How It Works

```
Input Text
    │
    ▼
┌─────────────────────────────────┐
│  Emotion Detection              │
│  j-hartmann/distilroberta-base  │
│  → label + confidence score     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Intensity Scaling              │
│  voice_params scaled by         │
│  confidence (0.3 → 1.0 range)   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  VITS Neural TTS                │
│  kakao-enterprise/vits-ljs      │
│  speaking_rate + noise_scale    │
│  set via model.config           │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Pitch Shift (post-processing)  │
│  torchaudio.functional          │
│  .pitch_shift()                 │
└─────────────────────────────────┘
    │
    ▼
  .wav output file + audio playback
```

---

## Emotion-to-Voice Mapping

Each of the 7 detected emotion labels maps to a distinct configuration of three
vocal parameters:

| Emotion  | Category | Speaking Rate | Noise Scale | Pitch Shift  |
|----------|----------|---------------|-------------|--------------|
| joy      | positive | 1.20x         | 0.500       | +2 semitones |
| surprise | positive | 1.30x         | 0.650       | +3 semitones |
| neutral  | neutral  | 1.00x         | 0.667       |  0 semitones |
| fear     | negative | 1.15x         | 0.750       | +1 semitone  |
| disgust  | negative | 0.90x         | 0.600       | -1 semitone  |
| anger    | negative | 1.10x         | 0.800       | -1 semitone  |
| sadness  | negative | 0.80x         | 0.400       | -2 semitones |

**Parameter explanations:**

- `speaking_rate` — controls how fast VITS stretches phoneme durations internally
- `noise_scale` — controls expressiveness and natural variation in the waveform
- `pitch_shift` — semitone shift applied post-synthesis via phase-vocoder (torchaudio)

---

## Intensity Scaling

A key design goal is **proportional modulation** — the degree of vocal change should
match the actual intensity of emotion in the text, not just its category.

This is implemented in `get_voice_params()` using the classifier's raw confidence score:

```python
scale         = max(0.3, min(1.0, confidence))
speaking_rate = neutral_rate  + scale * (emotion_rate  - neutral_rate)
noise_scale   = neutral_noise + scale * (emotion_noise - neutral_noise)
pitch_shift   = int(round(scale * emotion_pitch))
```

- Confidence `1.0` → full modulation (e.g., pitch +2 semitones)
- Confidence `0.5` → half modulation (e.g., pitch +1 semitone)
- Floor of `0.3` prevents completely flat output even on uncertain predictions

**Example:**

| Input                                    | Confidence | Rate   | Pitch  |
|------------------------------------------|------------|--------|--------|
| `"This is good."`                        | 0.55       | 1.11x  | +1st   |
| `"This is the best day of my LIFE!"`     | 0.97       | 1.19x  | +2st   |

---

## Code Structure

```
empathy-engine/
│
├── main.py              # Complete pipeline — single file
├── empathy_output/      # Auto-created at runtime; stores .wav files
├── requirements.txt     # Python dependencies
└── README.md
```

### main.py — Internal Layout

The file is organised into six strict layers, each with a single responsibility:

```
main.py
│
├── IMPORTS
│     Ordered by ascending line character length.
│
├── class Config
│   ├── VITS_MODEL_ID              HuggingFace model ID for VITS TTS
│   ├── EMOTION_MODEL_ID           HuggingFace model ID for emotion classifier
│   ├── OUTPUT_DIR                 Folder path for generated audio files
│   ├── EMOTION_VOICE_MAP          Core table: emotion label → vocal params
│   ├── EMOTION_CATEGORY_MAP       Maps 7 labels → positive / neutral / negative
│   └── __init__()                 Runtime GPU detection + VRAM logging
│
├── HTML_TEMPLATE
│     Single string containing the entire web UI (HTML + CSS + JavaScript).
│     Served by FastAPI at the root route. No separate files needed.
│
├── UTILITY FUNCTIONS
│   ├── log_message()              Timestamped [LEVEL] console logger
│   ├── setup_dependencies()       Installs espeak-ng via apt + phonemizer/fastapi
│   │                              via pip, then patches transformers module-level
│   │                              _phonemizer_available cache flag
│   ├── setup_output_dir()         Creates output folder if it does not exist
│   ├── apply_pitch_shift()        Wraps torchaudio.functional.pitch_shift safely
│   └── save_audio()               Normalises tensor dimensions and writes .wav
│
├── DATA / MODEL FUNCTIONS
│   ├── load_models()              Downloads and loads emotion pipeline + VITS
│   ├── detect_emotion()           Runs classifier, handles both dict and list
│   │                              output formats from transformers pipeline
│   ├── get_voice_params()         Fetches base params from Config and applies
│   │                              confidence-based intensity scaling
│   └── synthesize_speech()        Writes params into vits_model.config,
│                                  runs inference, applies pitch shift
│
├── INFERENCE FUNCTION
│   └── run_inference()            Orchestrates the full pipeline:
│                                  detect → scale → synthesize → save → return dict
│
├── WEB FUNCTION
│   └── create_web_app()           Builds FastAPI app with three routes:
│                                  GET /           → HTML UI
│                                  POST /synthesize → JSON result + audio URL
│                                  GET /audio/{f}  → streams .wav file
│
└── main()
    ├── setup_dependencies()       Always runs first
    ├── Config()                   Detects GPU at instantiation time
    ├── load_models()              Loads both models onto correct device
    ├── if --web flag              Starts uvicorn server on port 7860
    └── else                       Enters interactive CLI input loop
```

---

## Tech Stack

| Component         | Library / Model                                 |
|-------------------|-------------------------------------------------|
| Emotion Detection | `j-hartmann/emotion-english-distilroberta-base` |
| Neural TTS        | `kakao-enterprise/vits-ljs` (VITS, LJSpeech)    |
| Pitch Shifting    | `torchaudio.functional.pitch_shift`             |
| Model Framework   | HuggingFace `transformers` + `torch`            |
| Audio I/O         | `torchaudio`                                    |
| Web Framework     | `FastAPI` + `uvicorn`                           |
| Frontend          | Vanilla HTML/CSS/JS (inline, no build step)     |
| Phonemization     | `phonemizer` + `espeak-ng` (system binary)      |

---

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU recommended (CPU works, slower)
- Google Colab: select **Runtime → Change runtime type → T4 GPU**

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/empathy-engine.git
cd empathy-engine
```

### 2. Install Python Dependencies

```bash
pip install torch torchaudio transformers fastapi uvicorn python-multipart phonemizer
```

### 3. Install System Dependency

**Linux / Google Colab:**

```bash
apt-get install -y espeak-ng
```

**macOS:**

```bash
brew install espeak-ng
```

**Windows:**

Download and install from https://github.com/espeak-ng/espeak-ng/releases

> **Note:** Steps 2 and 3 are optional on fresh Colab sessions. The script's
> `setup_dependencies()` function automatically handles both installations at
> runtime and patches the transformers availability cache.

---

## Running the App

### Interactive CLI Mode

```bash
python main.py
```

You will be prompted to enter sentences one at a time:

```
🎙️  Empathy Engine — Interactive Mode
Type a sentence and press Enter. Type 'quit' to exit.

Enter sentence: I just got promoted! This is the best day of my life!
```

Expected output:

```
----------------------------------------------------------------------
  Text     : I just got promoted! This is the best day of my life!
  Emotion  : joy [positive] | conf: 0.9650
  Rate     : 1.197x
  Noise    : 0.503
  Pitch    : +2 semitones
  Output   : ./empathy_output/joy_20260223_214501.wav
----------------------------------------------------------------------
```

Type `quit`, `exit`, or `q` to stop. Press `Ctrl+C` to force exit.

### Web UI Mode

```bash
python main.py --web
```

Open your browser at:

```
http://localhost:7860
```

**On Google Colab**, expose the port using ngrok:

```python
!pip install pyngrok -q
from pyngrok import ngrok
public_url = ngrok.connect(7860)
print(public_url)
```

---

## Web Interface

The web UI is a single self-contained HTML page served by FastAPI at `/`.
There are no separate template files, no static asset folders, and no frontend
build toolchain required.

**Features:**

- Text area for sentence input
- Synthesize button that POSTs to `/synthesize`
- Live emotion tag with colour coding
  - Green for positive, grey for neutral, red for negative
- Results table showing emotion, confidence, rate, pitch, and noise scale
- Embedded audio player that auto-loads and plays the generated `.wav` immediately

**API Endpoints:**

| Method | Route              | Description                                       |
|--------|--------------------|---------------------------------------------------|
| GET    | `/`                | Serves the full web UI                            |
| POST   | `/synthesize`      | Accepts `{ "text": "..." }`, returns JSON result  |
| GET    | `/audio/{filename}`| Streams the generated `.wav` file to the browser  |

---

## Example Output

| Input Sentence                                  | Detected Emotion  | Rate   | Pitch |
|-------------------------------------------------|-------------------|--------|-------|
| `I just got promoted!`                          | joy [positive]    | 1.20x  | +2st  |
| `Nobody is helping me. This is unacceptable!`   | anger [negative]  | 1.10x  | -1st  |
| `Your account balance is one thousand dollars.` | neutral [neutral] | 1.00x  |  0st  |
| `I cannot believe this happened...`             | sadness [negative]| 0.80x  | -2st  |
| `Wait — did that actually just work?`           | surprise[positive]| 1.30x  | +3st  |

---

## Design Choices

### Why VITS over pyttsx3 or gTTS?

VITS is a fully neural end-to-end TTS model trained on LJSpeech that produces
significantly more natural speech than parametric engines like `pyttsx3`. Unlike
API-based options such as gTTS or ElevenLabs, it runs fully offline with no rate
limits, no API keys, and no latency from network calls. Crucially, it exposes
`speaking_rate` and `noise_scale` directly in its config, making programmatic
vocal modulation clean and precise.

### Why vits-ljs (single speaker)?

Consistent voice identity is central to the Empathy Engine's purpose. The listener
should hear the same voice expressing different emotions, not different speakers.
Multi-speaker models like `vits-vctk` would conflate emotional modulation with
speaker identity changes, muddying the perceptual effect.

### Why confidence-based intensity scaling?

A binary mapping treats `"okay"` and `"ABSOLUTELY INCREDIBLE"` identically if both
classify as `joy`. Scaling modulation by the classifier's raw confidence score
naturally handles this — stronger signals produce stronger vocal changes, directly
mirroring how humans modulate their voice in proportion to felt emotion.

### Why three vocal parameters?

- `speaking_rate` is the most perceptually salient cue (fast = excited or angry,
  slow = sad or calm)
- `noise_scale` controls naturalness and breath variation, changing how effortful
  or relaxed the voice sounds
- `pitch_shift` adds tonal height change, the most direct acoustic correlate of
  emotional arousal and valence in speech

### Why inline HTML_TEMPLATE?

Keeping the entire UI as a string inside `main.py` means the project deploys as a
true single-file application. No `templates/` folder, no static asset pipeline, no
npm. The full product — models, API, and frontend — is contained in one Python file.
