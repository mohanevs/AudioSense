"""
Live Audio Streaming and Continuous Microphone Loop for AudioSense.
Requirement 10: audiosense.loop()
"""

import time
import sys
from typing import Optional, Callable, Any
import numpy as np

def run_mic_loop(
    engine,
    chunk_duration: float = 3.0,
    sample_rate: int = 16000,
    device: Optional[int] = None,
    callback: Optional[Callable[[Any], None]] = None
):
    """
    Continuous microphone listening loop.
    Captures live audio from laptop mic, preprocesses, and continuously outputs
    transcriptions (if human speech) or environmental context statements (and updates DB).

    Parameters
    ----------
    engine : AudioSense
        AudioSense framework instance.
    chunk_duration : float, default 3.0
        Duration of each listening window in seconds.
    sample_rate : int, default 16000
        Microphone sampling rate.
    device : int, optional
        Microphone input device index.
    callback : callable, optional
        Optional function called with each result object.
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError(
            "sounddevice library is required for live mic listening. "
            "Install with: pip install sounddevice"
        )

    print("\n" + "=" * 65)
    print("      🎙️  AUDIOSENSE LIVE MICROPHONE LISTENING ACTIVE")
    print(f" Chunk Duration: {chunk_duration}s | Sample Rate: {sample_rate}Hz")
    print(f" Language Mode: {engine.language.upper()}")
    print(" Press Ctrl+C to stop listening.")
    print("=" * 65 + "\n")

    num_samples = int(chunk_duration * sample_rate)

    try:
        iteration = 1
        while True:
            t_start = time.time()
            sys.stdout.write(f"\r[Loop #{iteration}] Listening... ")
            sys.stdout.flush()

            # Record a chunk synchronously
            recording = sd.rec(
                num_samples,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=device
            )
            sd.wait()

            audio_chunk = recording.flatten()

            # Skip pure silence
            energy = np.sqrt(np.mean(audio_chunk ** 2))
            if energy < 0.005:
                sys.stdout.write(f"\r[Loop #{iteration}] (Ambient silence)               \n")
                iteration += 1
                continue

            sys.stdout.write(f"\r[Loop #{iteration}] Processing audio...       \n")
            sys.stdout.flush()

            # Predict on the live buffer
            result = engine.predict(audio_chunk, sr=sample_rate)

            # Output result
            print(f"  • Route: {result.route.upper()}")
            if result.route == "transcription":
                print(f"  • Transcript: \"{result.text}\"")
            else:
                print(f"  • Context: \"{result.text}\"")
                top_strs = [f"{lbl} ({conf:.2f})" for lbl, conf in result.labels[:3]]
                print(f"  • Detected Sounds: {', '.join(top_strs)}")

            if callback:
                callback(result)

            print("-" * 50)
            iteration += 1

    except KeyboardInterrupt:
        print("\n\n🛑 Live listening loop stopped by user.")
    except Exception as e:
        print(f"\n❌ Error during live microphone loop: {e}")
