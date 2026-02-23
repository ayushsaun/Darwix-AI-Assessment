import gc
import os
import sys
import time
import torch
import warnings
import subprocess
import torchaudio
from transformers import pipeline
import torchaudio.functional as AF
from transformers import VitsModel, AutoTokenizer


warnings.filterwarnings("ignore")


def log_message(message, level="INFO"):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {message}")


class Config:
    VITS_MODEL_ID    = "kakao-enterprise/vits-ljs"
    EMOTION_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
    OUTPUT_DIR       = "./empathy_output"

    EMOTION_VOICE_MAP = {
        "joy":      {"speaking_rate": 1.20, "noise_scale": 0.500, "pitch_shift":  2},
        "surprise": {"speaking_rate": 1.30, "noise_scale": 0.650, "pitch_shift":  3},
        "neutral":  {"speaking_rate": 1.00, "noise_scale": 0.667, "pitch_shift":  0},
        "fear":     {"speaking_rate": 1.15, "noise_scale": 0.750, "pitch_shift":  1},
        "disgust":  {"speaking_rate": 0.90, "noise_scale": 0.600, "pitch_shift": -1},
        "anger":    {"speaking_rate": 1.10, "noise_scale": 0.800, "pitch_shift": -1},
        "sadness":  {"speaking_rate": 0.80, "noise_scale": 0.400, "pitch_shift": -2},
    }

    EMOTION_CATEGORY_MAP = {
        "joy":      "positive",
        "surprise": "positive",
        "neutral":  "neutral",
        "fear":     "negative",
        "sadness":  "negative",
        "anger":    "negative",
        "disgust":  "negative",
    }

    def __init__(self):
        if torch.cuda.is_available():
            self.DEVICE          = torch.device("cuda:0")
            self.PIPELINE_DEVICE = 0
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            log_message(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")
        else:
            self.DEVICE          = torch.device("cpu")
            self.PIPELINE_DEVICE = -1
            log_message("CPU mode | Colab GPU: Runtime → Change runtime type → T4 GPU", "WARN")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Empathy Engine</title>
  <style>
    body     { font-family: Arial, sans-serif; max-width: 740px; margin: 60px auto; padding: 0 24px; background: #f4f6f8; }
    h1       { color: #2c3e50; margin-bottom: 4px; }
    p        { color: #555; margin-top: 0; }
    textarea { width: 100%; padding: 12px; font-size: 15px; border-radius: 8px; border: 1px solid #ccc; resize: vertical; box-sizing: border-box; }
    button   { margin-top: 12px; padding: 11px 32px; font-size: 15px; background: #2c3e50; color: #fff; border: none; border-radius: 8px; cursor: pointer; }
    button:hover { background: #3d5166; }
    #info    { margin-top: 22px; background: #fff; padding: 18px 20px; border-radius: 10px; border-left: 5px solid #2c3e50; display: none; }
    table    { border-collapse: collapse; width: 100%; }
    td, th   { padding: 7px 14px; text-align: left; border-bottom: 1px solid #f0f0f0; font-size: 14px; }
    th       { color: #888; font-weight: normal; width: 140px; }
    audio    { margin-top: 18px; width: 100%; display: none; }
    .tag     { display: inline-block; padding: 3px 11px; border-radius: 12px; font-size: 13px; font-weight: bold; }
    .positive{ background: #d4edda; color: #155724; }
    .neutral { background: #e2e3e5; color: #383d41; }
    .negative{ background: #f8d7da; color: #721c24; }
  </style>
</head>
<body>
  <h1>🎙️ Empathy Engine</h1>
  <p>Type a sentence and hear it spoken with emotionally-matched voice parameters.</p>
  <textarea id="txt" rows="4" placeholder="e.g. I just got promoted! This is the best day of my life!"></textarea>
  <br>
  <button onclick="synthesize()">▶ Synthesize</button>
  <div id="info">
    <table id="tbl"></table>
    <audio id="player" controls></audio>
  </div>
  <script>
    async function synthesize() {
      const text = document.getElementById('txt').value.trim();
      if (!text) return;
      const btn = document.querySelector('button');
      btn.innerText = 'Processing…';
      btn.disabled  = true;
      try {
        const resp = await fetch('/synthesize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });
        const d = await resp.json();
        const sign = d.voice_params.pitch_shift >= 0 ? '+' : '';
        const rows = [
          ['Emotion',     `<span class="tag ${d.emotion_category}">${d.emotion_label}</span> &nbsp;[${d.emotion_category}]`],
          ['Confidence',  (d.confidence * 100).toFixed(1) + '%'],
          ['Rate',        d.voice_params.speaking_rate + 'x'],
          ['Pitch',       sign + d.voice_params.pitch_shift + ' semitones'],
          ['Noise Scale', d.voice_params.noise_scale],
        ];
        document.getElementById('tbl').innerHTML = rows.map(
          ([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`
        ).join('');
        document.getElementById('info').style.display = 'block';
        const player = document.getElementById('player');
        player.src   = d.audio_url + '?t=' + Date.now();
        player.style.display = 'block';
        player.load();
        player.play();
      } finally {
        btn.innerText = '▶ Synthesize';
        btn.disabled  = false;
      }
    }
  </script>
</body>
</html>"""


def setup_dependencies():
    log_message("Installing espeak-ng system binary...")
    subprocess.run(["apt-get", "install", "-y", "espeak-ng"], capture_output=True, check=False)
    log_message("Installing Python packages...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "phonemizer", "fastapi", "uvicorn", "python-multipart", "-q"],
        capture_output=True, check=False
    )
    try:
        import importlib.util
        import transformers.utils.import_utils as _tiu
        _tiu._phonemizer_available = importlib.util.find_spec("phonemizer") is not None
        log_message(f"Transformers phonemizer cache patched → {_tiu._phonemizer_available}")
    except Exception as e:
        log_message(f"Cache patch skipped (non-critical): {e}", "WARN")


def setup_output_dir(path):
    os.makedirs(path, exist_ok=True)


def apply_pitch_shift(waveform, sample_rate, n_steps):
    if n_steps == 0:
        return waveform
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return AF.pitch_shift(waveform, sample_rate, n_steps)


def save_audio(waveform, output_path, sample_rate):
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(output_path, waveform.cpu().float(), sample_rate)


def load_models(config):
    log_message("Loading emotion detection model...")
    emotion_pipe = pipeline(
        "text-classification",
        model=config.EMOTION_MODEL_ID,
        top_k=None,
        device=config.PIPELINE_DEVICE,
    )
    log_message("Emotion model ready.")
    log_message("Loading VITS TTS model...")
    tokenizer  = AutoTokenizer.from_pretrained(config.VITS_MODEL_ID)
    vits_model = VitsModel.from_pretrained(config.VITS_MODEL_ID).to(config.DEVICE)
    vits_model.eval()
    log_message("VITS model ready.")
    return emotion_pipe, vits_model, tokenizer


def detect_emotion(text, emotion_pipe, config):
    if not text.strip():
        return "neutral", "neutral", 1.0
    raw     = emotion_pipe(text)
    results = raw[0]
    if isinstance(results, dict):
        label = results["label"].lower()
        score = results["score"]
    else:
        top   = max(results, key=lambda x: x["score"])
        label = top["label"].lower()
        score = top["score"]
    category = config.EMOTION_CATEGORY_MAP.get(label, "neutral")
    return label, category, round(score, 4)


def get_voice_params(emotion_label, config, confidence=1.0):
    base    = dict(config.EMOTION_VOICE_MAP.get(emotion_label, config.EMOTION_VOICE_MAP["neutral"]))
    neutral = config.EMOTION_VOICE_MAP["neutral"]
    scale   = max(0.3, min(1.0, confidence))
    base["speaking_rate"] = round(neutral["speaking_rate"] + scale * (base["speaking_rate"] - neutral["speaking_rate"]), 3)
    base["noise_scale"]   = round(neutral["noise_scale"]   + scale * (base["noise_scale"]   - neutral["noise_scale"]),   3)
    base["pitch_shift"]   = int(round(scale * base["pitch_shift"]))
    return base


def synthesize_speech(text, vits_model, tokenizer, voice_params, config):
    vits_model.config.speaking_rate = voice_params["speaking_rate"]
    vits_model.config.noise_scale   = voice_params["noise_scale"]
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        output = vits_model(**inputs)
    waveform    = output.waveform.squeeze().cpu()
    sample_rate = vits_model.config.sampling_rate
    waveform    = apply_pitch_shift(waveform, sample_rate, voice_params["pitch_shift"])
    return waveform, sample_rate


def run_inference(text, models, config):
    emotion_pipe, vits_model, tokenizer = models
    setup_output_dir(config.OUTPUT_DIR)
    log_message(f"Input    → {text[:80]}")
    emotion_label, emotion_category, confidence = detect_emotion(text, emotion_pipe, config)
    log_message(f"Emotion  → {emotion_label} [{emotion_category}] conf: {confidence:.4f}")
    voice_params = get_voice_params(emotion_label, config, confidence)          # ← confidence passed
    log_message(
        f"Params   → rate={voice_params['speaking_rate']}x | "
        f"noise={voice_params['noise_scale']} | "
        f"pitch={voice_params['pitch_shift']:+d}st"
    )
    waveform, sample_rate = synthesize_speech(text, vits_model, tokenizer, voice_params, config)
    filename    = f"{emotion_label}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    output_path = os.path.join(config.OUTPUT_DIR, filename)
    save_audio(waveform, output_path, sample_rate)
    log_message(f"Saved    → {output_path}")
    return {
        "text":             text,
        "emotion_label":    emotion_label,
        "emotion_category": emotion_category,
        "confidence":       confidence,
        "voice_params":     voice_params,
        "output_path":      output_path,
    }


def create_web_app(models, config):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="Empathy Engine")

    class SynthRequest(BaseModel):
        text: str

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_TEMPLATE

    @app.post("/synthesize")
    def synthesize(req: SynthRequest):
        result   = run_inference(req.text, models, config)
        filename = os.path.basename(result["output_path"])
        return JSONResponse({
            "emotion_label":    result["emotion_label"],
            "emotion_category": result["emotion_category"],
            "confidence":       result["confidence"],
            "voice_params":     result["voice_params"],
            "audio_url":        f"/audio/{filename}",
        })

    @app.get("/audio/{filename}")
    def serve_audio(filename: str):
        path = os.path.abspath(os.path.join(config.OUTPUT_DIR, filename))
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        return FileResponse(path, media_type="audio/wav")

    return app


def main():
    setup_dependencies()
    config = Config()
    models = load_models(config)

    if "--web" in sys.argv:
        import uvicorn
        app = create_web_app(models, config)
        log_message("Web UI → http://0.0.0.0:7860  (Colab: use ngrok or the Colab port link)")
        uvicorn.run(app, host="0.0.0.0", port=7860)
        return

    divider = "-" * 70
    print("\n🎙️  Empathy Engine — Interactive Mode")
    print("Type a sentence and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            text = input("Enter sentence: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not text or text.lower() in ("quit", "exit", "q"):
            break
        result = run_inference(text, models, config)
        print(divider)
        print(f"  Text     : {result['text'][:68]}")
        print(f"  Emotion  : {result['emotion_label']} [{result['emotion_category']}] | conf: {result['confidence']:.4f}")
        print(f"  Rate     : {result['voice_params']['speaking_rate']}x")
        print(f"  Noise    : {result['voice_params']['noise_scale']}")
        print(f"  Pitch    : {result['voice_params']['pitch_shift']:+d} semitones")
        print(f"  Output   : {result['output_path']}")
        print(divider)
        gc.collect()


if __name__ == "__main__":
    main()
