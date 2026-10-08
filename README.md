<div align="center">

# 🎙️ AudioSense SDK
### *Audio Intelligence for Machines, Robots, and Modern Applications*

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Version](https://img.shields.io/badge/version-1.0.0-00C49F.svg?style=for-the-badge)](https://github.com)
[![Models](https://img.shields.io/badge/Ensemble-AST%20%7C%20PANN%20%7C%20YAMNet-9900EF.svg?style=for-the-badge)](https://github.com)
[![Speech](https://img.shields.io/badge/Speech-Whisper%20%7C%20Silero%20VAD-FF6F00.svg?style=for-the-badge)](https://github.com)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg?style=for-the-badge)](https://github.com)
[![License](https://img.shields.io/badge/license-MIT-blueviolet.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-core-api-guide">Core API</a> •
  <a href="#-7-layer-architecture">Architecture</a> •
  <a href="#-live-streaming">Live Streaming</a>
</p>

---

</div>

## 📌 Overview

Computers see and read well, but most still hear poorly. A security camera sees a broken window; a microphone beside it usually outputs nothing but a raw waveform. 

**AudioSense** bridges that gap. It is a comprehensive audio intelligence framework that transforms raw acoustic signals into:
1. **Events** — *What happened and when* (sound classification & detection).
2. **Speech Transcripts** — *What was spoken* (automatic VAD gating and transcription).
3. **Insights & Context** — *What is happening in the environment* (LLM-driven situational reasoning).

Designed with a clean, tiered API: add intelligent hearing to your application or robot with **three lines of Python**, while retaining deep control over every layer underneath.

```python
from audiosense import AudioSense

model = AudioSense()
result = model.predict("environment.wav")
print(result)
# Output: Surroundings currently exhibit sounds of Water tap, faucet, Water, Sink (filling or washing).
```

---

## ✨ Key Features

| Capability | Description |
|---|---|
| **Multi-Model Ensemble** | Combines **AST** (Audio Spectrogram Transformer), **PANNs** (CNN14), and **YAMNet** in a single pass with calibrated consensus scoring. |
| **Intelligent Routing (Switcher)** | Automatically routes human speech to **Faster-Whisper** transcription and environmental sounds to **Context Intelligence / RAG**. |
| **Universal Audio Converter** | Ingests `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.opus`, `.webm`, `.mp4` and converts to pristine `.wav`. |
| **Dual Sample-Rate Conditioning** | Internal preprocessing pipeline delivers synchronized 32 kHz (for PANN) and 16 kHz (for AST/YAMNet/Whisper) audio arrays. |
| **Context Reasoning & LLM** | Synthesizes high-level environmental situation reports via local LLMs (Phi-4, llama-server, Ollama) or custom endpoints. |
| **Multilingual Translation** | Live translation of context statements and speech transcripts into Hindi (`'hin'`), Spanish, French, German, etc. |
| **Acoustic Database & Audit** | Built-in JSON event store tracking occurrence counts, timestamps, and confidence histories with `view_db()` and `summary_db()`. |
| **Real-time Live Mic Loop** | Continuous laptop microphone listening loop (`audiosense.loop()`) with VAD activity filtering and instant callbacks. |
| **Zero Log Pollution** | Completely silences noisy TensorFlow, oneDNN, and progress-bar outputs, giving clean, predictable returns. |

---

## 🚀 Installation

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-org/AudioSense.git
cd AudioSense
python -m venv env
# Windows
.\env\Scripts\activate
# Linux/macOS
source env/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Acquire Pre-Trained Weights
AudioSense dynamically downloads and verifies model weights:
```bash
# Downloads AST, PANN (CNN14), and YAMNet
python load_model.py

# Optional: Downloads local Phi-4-mini LLM (~2.5 GB)
python load_llm.py
```

---

## ⚡ Quickstart

```python
from audiosense import AudioSense, view_db

# 1. Initialize engine
model = AudioSense(device="auto", language="en")

# 2. Predict on an audio file
result = model.predict("audio/audio.wav")
print("Detected Environment:", result.text)
print("Top Labels:", result.labels)

# 3. View persistent sound history
view_db()
```

---

## 📖 Core API Guide

### 1. Unified Prediction & Routing
The switcher automatically checks whether incoming sound is human speech or environmental:
```python
res = model.predict("audio/sample.wav")

if res.route == "transcription":
    print("Speech detected:", res.transcription)
else:
    print("Context statement:", res.context)
```

### 2. Functional Label APIs
Direct functional access to taxonomy and model predictions:
```python
from audiosense.label import get_label, get_all, get_categories, top_k

# Primary label
primary = get_label("audio/audio.wav")

# Top 5 consensus predictions
rankings = top_k("audio/audio.wav", k=5)

# Inspect categories
cats = get_categories(["Dog", "Car horn", "Speech"])
# {'Dog': 'Animal sounds', 'Car horn': 'Sounds of things', 'Speech': 'Human sounds'}

# Full predictions across all models
all_preds = get_all("audio/audio.wav", threshold=0.1)
```

### 3. Switcher Tuning
Customize routing rules dynamically for your specific application:
```python
# Move bird chirping to human/speech indicators for this instance
model.switcher("Birds_chirping", "HUMAN")

# Or move a sound to environmental indicators
model.switcher("Shout", "ENV")
```

### 4. Custom LLM Integration
Plug in your own local or remote language model:
```python
# Custom GGUF file path
model.llm(path="C:/models/custom_model.gguf")

# Or custom Ollama / llama-server HTTP endpoint
model.llm(url="http://localhost:11434/api/generate")
```

### 5. Multilingual Output
Output context statements and transcripts in your desired language:
```python
model.translation('hin')  # Output in Hindi
res = model.predict("audio/audio.wav")
print(res.text)  # e.g., 'आस-पास पानी के नल और सिंक की आवाजें आ रही हैं।'
```

### 6. Database Inspection & Summaries
```python
# Print formatted database table
model.view_db()

# Generate executive summary of auditory history using LLM
model.summary_db()
```

### 7. Universal Audio Conversion
```python
from audiosense import convert_to_wav

# Accepts MP3, FLAC, M4A, OGG, WEBM, MP4, etc.
wav_path = convert_to_wav("recording.m4a", output_path="recording.wav")
```

---

## 🎙️ Live Microphone & Robotics

Continuous, non-blocking real-time listening through the system microphone:

```python
from audiosense import AudioSense

model = AudioSense()

# Start continuous listening loop (Press Ctrl+C to stop)
model.loop(chunk_duration=3.0)
```

You can also provide a callback for robot reactive control:
```python
def on_sound(res):
    if "Siren" in [lbl for lbl, _ in res.labels]:
        print("🚨 Warning: Emergency siren heard! Halting robot.")

model.loop(chunk_duration=2.0, callback=on_sound)
```

---

## 🏛️ 7-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 7. Application & Robot Interface                            │
│    Python API (AudioSense) • Callbacks • Controller Maps    │
├─────────────────────────────────────────────────────────────┤
│ 6. Context Intelligence Layer                               │
│    Situational Reasoning • LLM Summaries • Multilingual     │
├─────────────────────────────────────────────────────────────┤
│ 5. Ensemble Decision Layer & Switcher                       │
│    Weighted Consensus • Speech vs. Environmental Router     │
├─────────────────────────────────────────────────────────────┤
│ 4. AI Model Layer                                           │
│    AST (Transformer) • PANN (CNN14) • YAMNet • Whisper      │
├─────────────────────────────────────────────────────────────┤
│ 3. Feature Extraction Layer                                 │
│    Log-mel Spectrograms • Filterbanks • Embeddings          │
├─────────────────────────────────────────────────────────────┤
│ 2. Audio Processing Layer                                   │
│    Resampling (16 kHz / 32 kHz) • VAD Gating • Normalization│
├─────────────────────────────────────────────────────────────┤
│ 1. Audio Input Layer                                        │
│    File Decoders (conversion.py) • Microphones • Streams    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Project Structure

```
AudioSense/
├── audiosense/
│   ├── __init__.py           # Public framework exports
│   ├── core/
│   │   ├── engine.py         # AudioSense primary engine
│   │   ├── config.py         # Dynamic relative path resolver
│   │   └── silence.py        # Log & warning suppression engine
│   ├── audio/
│   │   ├── conversion.py     # Universal audio converter (mp3/flac/m4a -> wav)
│   │   ├── preprocess.py     # Dual-rate (16k/32k) conditioning
│   │   └── io.py             # Audio reader/writer utilities
│   ├── models/
│   │   ├── ast_model.py      # AST adapter
│   │   ├── pann_model.py     # PANN (CNN14) adapter
│   │   ├── yamnet_model.py   # YAMNet adapter
│   │   ├── whisper_model.py  # Faster-Whisper adapter
│   │   └── manager.py        # Lazy model coordinator
│   ├── label.py & labels/    # Functional label APIs & AudioSet taxonomy
│   ├── switcher/
│   │   ├── router.py         # Tunable Speech vs Environment Router
│   │   └── vad.py            # Silero-VAD detector
│   ├── context/
│   │   ├── llm.py            # LLM interface (Ollama / llama-server / custom)
│   │   ├── translator.py     # Multi-language translation engine
│   │   └── generator.py      # Context & database summarizers
│   ├── storage/
│   │   └── database.py       # JSON database engine (view_db)
│   └── live/
│       └── loop.py           # Real-time microphone listening loop
├── models/                   # Local model weights cache
├── load_model.py             # Model acquisition script
├── load_llm.py               # LLM downloader script
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
