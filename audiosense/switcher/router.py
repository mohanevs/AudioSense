"""
AudioSense Intelligent Switcher / Router.
Routes audio to TRANSCRIPTION if human speech is detected,
or to RAG / CONTEXT GENERATION if environmental sounds are detected.

Supports instance-level tuning:
e.g. switcher("Birds_chirping", "HUMAN") dynamically moves sounds
between ENV and HUMAN indicators.
"""

import re
from typing import Dict, List, Set, Any, Union, Optional

DEFAULT_SPEECH_INDICATORS: Set[str] = {
    "speech", "speaking", "speak", "spoken", "talk", "talking", "talks",
    "conversation", "conversing", "dialogue", "voice", "voices", "command",
    "commands", "instruction", "instructions", "verbal", "utterance", "utterances",
    "person speaking", "people talking", "people talk", "human speech", "human voice",
    "man speaking", "woman speaking", "male voice", "female voice", "male speech, man speaking",
    "female speech, woman speaking", "child speech, kid speaking", "whispering", "whisper",
    "shouting", "shout", "chatter", "crowd", "hubbub, speech noise, speech babble"
}

DEFAULT_NON_SPEECH_HUMAN_SOUNDS: Set[str] = {
    "laugh", "laughing", "laughter", "cry", "crying", "scream", "screaming",
    "breathing", "cough", "coughing", "sneeze", "sneezing", "snoring", "gasp",
    "groan", "grunt", "sigh", "whimper", "throat clearing", "clapping", "applause"
}


def normalize_label(text: str) -> str:
    """Normalize label string into lowercase, underscore/hyphen-free tokens."""
    t = str(text).lower().strip()
    t = t.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", t)


class Switcher:
    """
    Intelligent routing between Speech Transcription and Environmental RAG/Context.
    Tunable per instance.
    """
    def __init__(self):
        self.human_indicators: Set[str] = set(DEFAULT_SPEECH_INDICATORS)
        self.non_speech_human: Set[str] = set(DEFAULT_NON_SPEECH_HUMAN_SOUNDS)
        self.env_overrides: Set[str] = set()

    def move_label(self, label: str, target_category: str) -> str:
        """
        Move a sound label from environmental to human or vice-versa.

        Parameters
        ----------
        label : str
            Label name (e.g. "Birds_chirping")
        target_category : str
            "HUMAN" or "ENV" / "ENVIRONMENTAL"
        """
        norm = normalize_label(label)
        cat = target_category.upper().strip()

        if "HUMAN" in cat:
            self.human_indicators.add(norm)
            self.env_overrides.discard(norm)
            return f"Moved '{label}' to HUMAN indicators."
        elif "ENV" in cat:
            self.human_indicators.discard(norm)
            self.env_overrides.add(norm)
            return f"Moved '{label}' to ENVIRONMENTAL indicators."
        else:
            raise ValueError(f"Unknown target category: {target_category}. Use 'HUMAN' or 'ENV'.")

    # Callable shortcut for tuner: switcher("Birds_chirping", "HUMAN")
    def __call__(self, label: str, target_category: str) -> str:
        return self.move_label(label, target_category)

    def is_speech_label(self, label: str) -> bool:
        """Check if label indicates human speech."""
        norm = normalize_label(label)
        if norm in self.env_overrides:
            return False

        # Direct indicator check
        if norm in self.human_indicators:
            return True

        # Substring / token matching
        for ind in self.human_indicators:
            if ind in norm or norm in ind:
                return True

        return False

    def route(self, labels: List[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Accept any list of labels (or label-score dicts) and decide route:
        'transcription' or 'rag'
        """
        parsed_labels = []
        for item in labels:
            if isinstance(item, dict):
                lbl = item.get("label", "")
                conf = item.get("confidence", item.get("score", 1.0))
            elif isinstance(item, (tuple, list)):
                lbl, conf = item[0], item[1]
            else:
                lbl, conf = str(item), 1.0
            parsed_labels.append((lbl, conf))

        matched_speech = []
        for lbl, conf in parsed_labels:
            if self.is_speech_label(lbl):
                matched_speech.append({"label": lbl, "confidence": conf})

        if matched_speech:
            return {
                "route": "transcription",
                "reason": f"Detected speech-related sound: {matched_speech[0]['label']}",
                "speech_labels": matched_speech,
                "all_labels": [{"label": l, "confidence": c} for l, c in parsed_labels]
            }

        return {
            "route": "rag",
            "reason": "Environmental sounds detected (no speech indicator).",
            "speech_labels": [],
            "all_labels": [{"label": l, "confidence": c} for l, c in parsed_labels]
        }
