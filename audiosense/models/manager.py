"""
Model Manager coordinating AST, PANN, YAMNet, and Whisper models.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any

from .ast_model import ASTModel
from .pann_model import PANNModel
from .yamnet_model import YAMNetModel
from .whisper_model import WhisperTranscriber
from ..core.config import DEFAULT_MODELS_DIR

class ModelManager:
    def __init__(self, models_dir: Optional[Union[str, Path]] = None, device: str = "auto"):
        self.models_dir = Path(models_dir or DEFAULT_MODELS_DIR).resolve()
        self.device = device
        self._ast: Optional[ASTModel] = None
        self._pann: Optional[PANNModel] = None
        self._yamnet: Optional[YAMNetModel] = None
        self._whisper: Optional[WhisperTranscriber] = None

    @property
    def ast(self) -> ASTModel:
        if self._ast is None:
            self._ast = ASTModel(self.models_dir / "ast", device=self.device)
        return self._ast

    @property
    def pann(self) -> PANNModel:
        if self._pann is None:
            self._pann = PANNModel(self.models_dir / "pann" / "Cnn14_mAP=0.431.pth", device=self.device)
        return self._pann

    @property
    def yamnet(self) -> YAMNetModel:
        if self._yamnet is None:
            self._yamnet = YAMNetModel(self.models_dir / "yamnet")
        return self._yamnet

    @property
    def whisper(self) -> WhisperTranscriber:
        if self._whisper is None:
            self._whisper = WhisperTranscriber(device="cpu")
        return self._whisper
