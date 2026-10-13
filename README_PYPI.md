# AudioSense

Audio intelligence framework for machines, robots, and modern applications.

AudioSense provides a unified, multi-model audio perception library that transforms raw acoustic signals into sound events, speech transcripts, and contextual environmental insights.

The AudioSense framework includes the following:
- **Sound Event Classification**: Ensemble consensus scoring combining AST (Audio Spectrogram Transformer), PANNs (CNN14), and YAMNet.
- **Intelligent Routing**: Automatic VAD gating that routes human speech to Faster-Whisper and environmental audio to situational reasoning.
- **Context Reasoning**: High-level environmental situation descriptions powered by local LLMs (Ollama, llama-server, GGUF) or custom endpoints.
- **Universal Audio Ingestion**: Ingests `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.opus`, `.webm`, and `.mp4`, converting to conditioned `.wav`.
- **Live Real-time Microphone Loop**: Continuous listening with callback hooks for robotics and alert systems.
- **Acoustic Event History**: Built-in persistence for tracking auditory events over time with summary reporting.

## Installation

Install AudioSense using pip:

```bash
pip install audiosense
```

For optional server dependencies (FastAPI & Uvicorn):

```bash
pip install "audiosense[server]"
```

## Quickstart

Add intelligent audio perception to your application in a few lines of code:

```python
from audiosense import AudioSense

# Initialize engine
model = AudioSense()

# Predict on an audio file
result = model.predict("audio.wav")
print(result)

# Inspect detected labels and context
print("Detected labels:", result.labels)
print("Context:", result.text)
```

## Core Capabilities

### 1. Unified Speech vs. Environmental Routing
AudioSense automatically determines whether incoming sound is human speech or ambient environmental sound:

```python
res = model.predict("sample.wav")

if res.route == "transcription":
    print("Speech transcript:", res.transcription)
else:
    print("Environmental context:", res.context)
```

### 2. Functional Label APIs
Direct access to AudioSet taxonomy and multi-model consensus predictions:

```python
from audiosense.label import get_label, top_k, get_categories

# Top prediction
primary = get_label("sample.wav")

# Top 5 consensus predictions
rankings = top_k("sample.wav", k=5)

# Inspect categories
categories = get_categories(["Dog", "Car horn", "Speech"])
```

### 3. Real-Time Microphone Listening
Continuous listening loop for robots and interactive systems:

```python
from audiosense import AudioSense

model = AudioSense()

def on_sound(res):
    print("Heard:", res.labels)

# Start real-time listening loop (Press Ctrl+C to stop)
model.loop(chunk_duration=2.0, callback=on_sound)
```

### 4. Universal Audio Conversion
Convert any supported audio format to standardized `.wav`:

```python
from audiosense import convert_to_wav

wav_path = convert_to_wav("recording.m4a", output_path="recording.wav")
```

## Project Links

- **Source Code**: [https://github.com/mohanevs/AudioSense](https://github.com/mohanevs/AudioSense)
- **Issue Tracker**: [https://github.com/mohanevs/AudioSense/issues](https://github.com/mohanevs/AudioSense/issues)

## Discussion & Support

If you run into bugs or have feature suggestions, please file them in our [GitHub Issue Tracker](https://github.com/mohanevs/AudioSense/issues).

## License

AudioSense is licensed under the [MIT License](https://github.com/mohanevs/AudioSense/blob/main/LICENSE).
