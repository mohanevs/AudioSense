"""
AudioSense JSON Database Manager.
Maintains history of detected audio labels, counts, timestamps, and confidence scores.
Provides `view_db()` and utilities for RAG / LLM context generation.
"""

import json
import os
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

from ..core.config import DEFAULT_DB_PATH

def update_audio_labels(
    new_labels: Union[str, List[str]],
    confidences: Union[float, List[float]],
    file_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Updates the JSON database with newly detected labels and confidence scores.
    Moves previously 'current_labels' into 'old_labels'.
    """
    db_file = Path(file_path or DEFAULT_DB_PATH).resolve()
    db_file.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(new_labels, str):
        new_labels = [new_labels]
    if isinstance(confidences, (float, int)):
        confidences = [float(confidences)]

    if len(new_labels) != len(confidences):
        raise ValueError(
            f"Label count ({len(new_labels)}) must match confidence count ({len(confidences)})."
        )

    data = {"current_labels": {}, "old_labels": {}}
    if db_file.exists():
        try:
            with open(db_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # 1. Archive previous current_labels into old_labels
    old = data.setdefault("old_labels", {})
    for label, info in data.get("current_labels", {}).items():
        if label in old:
            old[label]["count"] += info["count"]
            old[label].setdefault("timestamps", []).extend(info.get("timestamps", []))
            old[label].setdefault("confidence_scores", []).extend(info.get("confidence_scores", []))
        else:
            old[label] = copy.deepcopy(info)

    # 2. Reset current_labels for the latest inference
    data["current_labels"] = {}
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for label, conf in zip(new_labels, confidences):
        lbl = str(label).strip()
        conf = float(conf)
        if lbl not in data["current_labels"]:
            data["current_labels"][lbl] = {
                "count": 0,
                "timestamps": [],
                "confidence_scores": []
            }
        data["current_labels"][lbl]["count"] += 1
        data["current_labels"][lbl]["timestamps"].append(current_time)
        data["current_labels"][lbl]["confidence_scores"].append(round(conf, 4))

    # 3. Write back atomically to disk
    with open(db_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return data


def view_db(file_path: Optional[Union[str, Path]] = None, return_str: bool = False) -> Any:
    """
    Displays and returns the current JSON database contents.
    Requirement 6: view_db()
    """
    db_file = Path(file_path or DEFAULT_DB_PATH).resolve()
    if not db_file.exists():
        msg = f"[AudioSense DB] Database file not found at: {db_file}"
        print(msg)
        return {"current_labels": {}, "old_labels": {}} if not return_str else msg

    with open(db_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build human-readable formatted representation
    lines = [
        "=" * 65,
        f"             AUDIOSENSE DATABASE OVERVIEW",
        f" Path: {db_file}",
        "=" * 65,
        "\n--- CURRENT ACTIVE SOUNDS ---"
    ]

    curr = data.get("current_labels", {})
    if not curr:
        lines.append("  (No active sound detected in current cycle)")
    else:
        for lbl, info in curr.items():
            last_ts = info["timestamps"][-1] if info["timestamps"] else "N/A"
            avg_conf = (
                sum(info["confidence_scores"]) / len(info["confidence_scores"])
                if info["confidence_scores"] else 0.0
            )
            lines.append(f"  • {lbl:<25} | Conf: {avg_conf:.2f} | Time: {last_ts}")

    lines.append("\n--- HISTORICAL SOUND LOGS ---")
    old = data.get("old_labels", {})
    if not old:
        lines.append("  (No historical sound logs recorded yet)")
    else:
        for lbl, info in sorted(old.items(), key=lambda x: x[1].get("count", 0), reverse=True):
            count = info.get("count", 0)
            scores = info.get("confidence_scores", [])
            avg_conf = sum(scores) / len(scores) if scores else 0.0
            lines.append(f"  • {lbl:<25} | Repeated: {count:<4} times | Avg Conf: {avg_conf:.2f}")

    lines.append("=" * 65)
    formatted = "\n".join(lines)
    print(formatted)

    return formatted if return_str else data
