"""
AudioSense LLM Model Downloader.
Handles downloading local Phi-4-mini GGUF weights into the models directory.
"""

import os
from pathlib import Path
from typing import Optional

MODEL_FILENAME = "Phi-4-mini-instruct-Q4_K_M.gguf"
MODEL_URL = (
    "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF"
    "/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
)
MIN_VALID_SIZE = 1_000_000_000


def download_llm_model(llm_dir: Optional[Path] = None) -> Path:
    from ..core.config import DEFAULT_MODELS_DIR
    target_dir = Path(llm_dir or (DEFAULT_MODELS_DIR / "llm")).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / MODEL_FILENAME

    if dest.exists() and dest.stat().st_size >= MIN_VALID_SIZE:
        print(f"[LLM] Model already exists at: {dest}")
        return dest

    print(f"[LLM] Downloading Phi-4-mini-instruct (~2.5 GB) to {target_dir}...")
    import requests

    part = dest.with_suffix(dest.suffix + ".part")
    resume_from = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with requests.get(MODEL_URL, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        total = resume_from + int(r.headers.get("content-length", 0))
        done = resume_from
        with open(part, "ab" if resume_from else "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
                done += len(block)
                if total:
                    print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({done * 100 / total:5.1f}%)", end="", flush=True)
        print()

    part.replace(dest)
    print(f"[LLM] Model saved to {dest}")
    return dest
