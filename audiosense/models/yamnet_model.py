"""
YAMNet Model Adapter for AudioSense.
"""

import os
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np

from ..core.config import DEFAULT_MODELS_DIR
from ..core.silence import silence_all

class YAMNetModel:
    def __init__(self, model_dir: Optional[Union[str, Path]] = None):
        self.model_dir = Path(model_dir or (DEFAULT_MODELS_DIR / "yamnet")).resolve()
        self._model = None
        self._class_names = None

    def _load(self):
        if self._model is not None:
            return

        with silence_all():
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
            import tensorflow as tf
            import tensorflow_hub as hub

            tf.get_logger().setLevel("ERROR")
            tf.autograph.set_verbosity(0)
            if hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
                tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

            if not self.model_dir.exists():
                raise FileNotFoundError(
                    f"YAMNet model directory not found at {self.model_dir}. "
                    "Run `python load_model.py` or `audiosense.download_models()`."
                )

            self._model = hub.load(str(self.model_dir))
            class_map_path = self._model.class_map_path().numpy().decode("utf-8")

            names = []
            with tf.io.gfile.GFile(class_map_path) as f:
                reader = csv.reader(f)
                next(reader, None)  # header
                for row in reader:
                    if len(row) >= 3:
                        names.append(row[2])
            self._class_names = names

    def predict(self, audio_16k: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Run inference on 16 kHz mono float32 waveform.
        Returns list of (label, score) tuples.
        """
        self._load()
        with silence_all():
            scores, _, _ = self._model(audio_16k)
            mean_scores = scores.numpy().mean(axis=0)

        top_k = min(top_k, len(mean_scores))
        top_idx = np.argpartition(mean_scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(mean_scores[top_idx])[::-1]]
        return [(self._class_names[i], float(mean_scores[i])) for i in top_idx]

    def predict_all(self, audio_16k: np.ndarray, threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        Run inference and return all labels matching or exceeding threshold.
        """
        self._load()
        with silence_all():
            scores, _, _ = self._model(audio_16k)
            mean_scores = scores.numpy().mean(axis=0)

        order = np.argsort(-mean_scores)
        return [
            (self._class_names[i], float(mean_scores[i]))
            for i in order
            if i < len(self._class_names) and mean_scores[i] >= threshold
        ]
