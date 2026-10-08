"""
Audio Conversion Module for AudioSense.
Accepts any audio or video container format (MP3, FLAC, OGG, M4A, AAC, OPUS, WEBM, MP4, etc.)
and converts it to standardized .wav format for model processing.
"""

import os
import tempfile
from pathlib import Path
from typing import Union, Optional
import numpy as np

def convert_to_wav(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    target_sr: Optional[int] = None,
    mono: bool = True
) -> Path:
    """
    Converts any audio file to .wav format.

    Parameters
    ----------
    input_path : str | Path
        Path to the source audio/video file.
    output_path : str | Path, optional
        Destination path. If None, creates a file with '.wav' extension next to the source
        or in a temporary directory if source directory is not writable.
    target_sr : int, optional
        Target sample rate. If None, keeps original or defaults to standard 16000 or 32000.
    mono : bool, default True
        Whether to convert channels to single mono channel.

    Returns
    -------
    Path
        Absolute path to the resulting .wav file.
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    # Determine destination
    if output_path is None:
        if input_path.suffix.lower() == ".wav" and target_sr is None:
            return input_path
        output_path = input_path.with_suffix(".wav")
        # If output matches input, rename output
        if output_path == input_path:
            output_path = input_path.parent / f"{input_path.stem}_converted.wav"
    else:
        output_path = Path(output_path).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try librosa + soundfile (best for universal decoding)
    try:
        import librosa
        import soundfile as sf

        audio, sr = librosa.load(str(input_path), sr=target_sr, mono=mono)
        sf.write(str(output_path), audio, samplerate=sr, subtype="PCM_16")
        return output_path
    except Exception as e_librosa:
        # Fallback 1: soundfile directly
        try:
            import soundfile as sf
            data, sr = sf.read(str(input_path))
            if mono and data.ndim > 1:
                data = data.mean(axis=1)
            sf.write(str(output_path), data, samplerate=sr, subtype="PCM_16")
            return output_path
        except Exception:
            pass

        # Fallback 2: moviepy for media/video containers (e.g. mp4, webm)
        try:
            from moviepy.editor import AudioFileClip
            clip = AudioFileClip(str(input_path))
            clip.write_audiofile(
                str(output_path),
                fps=target_sr or 16000,
                nbytes=2,
                codec='pcm_s16le',
                ffmpeg_params=["-ac", "1"] if mono else None,
                logger=None
            )
            clip.close()
            return output_path
        except Exception:
            pass

        raise RuntimeError(
            f"Failed to convert {input_path} to WAV. Error: {e_librosa}"
        )

def is_audio_file(file_path: Union[str, Path]) -> bool:
    """Check if file has an audio or video extension known to contain sound."""
    supported_extensions = {
        ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac",
        ".opus", ".webm", ".wma", ".aiff", ".mp4", ".mkv"
    }
    return Path(file_path).suffix.lower() in supported_extensions
