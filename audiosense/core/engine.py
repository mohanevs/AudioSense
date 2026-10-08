"""
AudioSense Core Framework Engine.
The primary entry point of the AudioSense library.
"""

from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np

from .config import DEFAULT_MODELS_DIR, DEFAULT_DB_PATH
from .silence import silence_all
from ..audio.preprocess import preprocess_audio, ProcessedAudio
from ..models.manager import ModelManager
from ..labels.top import fuse_predictions, top_k
from ..labels.all import get_all
from ..switcher.router import Switcher
from ..switcher.vad import VADDetector
from ..context.llm import LLMClient
from ..context.generator import generate_context_statement, generate_database_summary
from ..context.translator import translate_text
from ..storage.database import update_audio_labels, view_db


@dataclass
class AudioSenseResult:
    """Rich result object returned by model.predict()."""
    route: str                                      # 'transcription' or 'context'
    text: str                                       # Primary result (context statement or transcription)
    labels: List[tuple]                             # Top consensus labels (label, score)
    all_predictions: Dict[str, List[tuple]]         # Predictions from ast, pann, yamnet
    transcription: Optional[str] = None             # Speech transcription if speech detected
    context: Optional[str] = None                   # Environmental statement if environmental
    language: str = "en"                            # Active language
    duration: float = 0.0

    def __repr__(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.text


class AudioSense:
    """
    AudioSense Framework Main Class.

    Usage
    -----
    >>> from audiosense import AudioSense
    >>> model = AudioSense()
    >>> result = model.predict("audio/audio.wav")
    >>> print(result)
    """

    def __init__(
        self,
        device: str = "auto",
        language: str = "en",
        models_dir: Optional[Union[str, Path]] = None,
        db_path: Optional[Union[str, Path]] = None,
        llm_path: Optional[Union[str, Path]] = None,
        llm_url: Optional[str] = None
    ):
        with silence_all():
            self.device = device
            self.language = language
            self.models_dir = Path(models_dir or DEFAULT_MODELS_DIR).resolve()
            self.db_path = Path(db_path or DEFAULT_DB_PATH).resolve()

            # Modular subsystems with lazy model loading
            self.models = ModelManager(models_dir=self.models_dir, device=self.device)
            self._switcher = Switcher()
            self.vad = VADDetector()
            self._llm = LLMClient(custom_path=llm_path, endpoint_url=llm_url)

    # Internal audio preprocessing
    def preprocess(
        self,
        audio: Union[str, Path, np.ndarray, ProcessedAudio],
        sr: Optional[int] = None
    ) -> ProcessedAudio:
        """
        Preprocesses audio into dual sample-rate representations:
        32 kHz for PANN and 16 kHz for AST, YAMNet, Whisper.
        """
        with silence_all():
            return preprocess_audio(audio, orig_sr=sr)

    # Dynamic switcher tuning per instance
    def switcher(self, label: str, target_category: str) -> str:
        """
        Move a sound label from environmental to human or vice-versa for this instance.
        Example: model.switcher("Birds_chirping", "HUMAN")
        """
        return self._switcher.move_label(label, target_category)

    # Custom LLM configuration
    def llm(
        self,
        path: Optional[Union[str, Path]] = None,
        url: Optional[str] = None
    ) -> str:
        """
        Configure a custom LLM path or remote URL endpoint.
        """
        self._llm.configure(path=path, url=url)
        return f"Configured LLM path: {self._llm.custom_path} (endpoint: {self._llm.endpoint_url})"

    # Language translation configuration
    def translation(self, lang_code: str) -> str:
        """
        Set target language code for output context statements and transcripts.
        Example: 'hin' for Hindi, 'es' for Spanish, 'en' for English.
        """
        self.language = lang_code.lower().strip()
        return f"AudioSense target translation language set to: {self.language}"

    # View database records
    def view_db(self, return_str: bool = False) -> Any:
        """
        Displays and returns the current JSON database.
        """
        return view_db(file_path=self.db_path, return_str=return_str)

    # Summary of database history using LLM
    def summary_db(self) -> str:
        """
        Generates a well-formatted summary of the database using the LLM
        (translated if language is set to 'hin' etc.).
        """
        data = view_db(file_path=self.db_path)
        with silence_all():
            summary = generate_database_summary(data, self._llm, language=self.language)
        print("\n" + summary)
        return summary

    # Audio prediction and classification
    def predict(
        self,
        audio: Union[str, Path, np.ndarray, ProcessedAudio],
        sr: Optional[int] = None,
        top_n: int = 5
    ) -> AudioSenseResult:
        """
        Predict sound events from audio, apply switcher routing, and return
        transcription or situation context statement.
        """
        with silence_all():
            # Preprocess audio
            proc: ProcessedAudio = self.preprocess(audio, sr=sr)

            # Extract predictions from each model (each runs once)
            ast_preds = self.models.ast.predict(proc.audio_16k, top_k=top_n)
            pann_preds = self.models.pann.predict(proc.audio_32k, top_k=top_n)
            yamnet_preds = self.models.yamnet.predict(proc.audio_16k, top_k=top_n)

            all_preds = {
                "ast": ast_preds,
                "pann": pann_preds,
                "yamnet": yamnet_preds
            }

            # Consensus top labels
            fused_labels = fuse_predictions(ast_preds, pann_preds, yamnet_preds, k=top_n)

            # Route via Switcher
            decision = self._switcher.route(fused_labels)
            route = decision["route"]

            transcription_text = None
            context_statement = None
            final_text = ""

            if route == "transcription":
                # Speech transcription with Whisper
                raw_transcription = self.models.whisper.transcribe(proc.audio_16k)
                # Translate if configured (e.g. 'hin')
                transcription_text = translate_text(
                    raw_transcription,
                    target_lang=self.language,
                    llm_client=self._llm
                )
                final_text = transcription_text
            else:
                # Environmental route: update database and generate context statement
                lbl_names = [lbl for lbl, _ in fused_labels]
                lbl_scores = [score for _, score in fused_labels]
                update_audio_labels(lbl_names, lbl_scores, file_path=self.db_path)

                labels_payload = [{"label": l, "confidence": s} for l, s in fused_labels]
                context_statement = generate_context_statement(
                    labels_payload,
                    llm_client=self._llm,
                    language=self.language
                )
                final_text = context_statement

        return AudioSenseResult(
            route=route,
            text=final_text,
            labels=fused_labels,
            all_predictions=all_preds,
            transcription=transcription_text,
            context=context_statement,
            language=self.language,
            duration=proc.duration
        )

    # Continuous microphone listening loop
    def loop(
        self,
        chunk_duration: float = 3.0,
        sample_rate: int = 16000,
        device: Optional[int] = None,
        callback: Optional[Any] = None
    ):
        """
        Starts continuous laptop microphone listening loop.
        """
        from ..live.loop import run_mic_loop
        run_mic_loop(
            engine=self,
            chunk_duration=chunk_duration,
            sample_rate=sample_rate,
            device=device,
            callback=callback
        )
