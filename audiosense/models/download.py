"""
AudioSense Model Downloader Module.
Handles downloading AST, PANN, and YAMNet weights into the active models directory.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

AST_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

PANN_FILES = {
    "Cnn14_mAP=0.431.pth":
        "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
    "class_labels_indices.csv":
        "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
}

YAMNET_HANDLE = "google/yamnet/tensorFlow2/yamnet"


def download_file(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Stream a file to disk with progress output."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)
                done += len(block)
                if total:
                    print(f"\r  {dest.name}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
    tmp.replace(dest)


def download_ast(target_dir: Path) -> None:
    """Download AST model files from HuggingFace."""
    if (target_dir / "config.json").exists() and (
        any(target_dir.glob("*.safetensors")) or any(target_dir.glob("*.bin"))
    ):
        return

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    print(f"[AST] Downloading weights to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(AST_NAME)
    model = AutoModelForAudioClassification.from_pretrained(AST_NAME)
    feature_extractor.save_pretrained(target_dir)
    model.save_pretrained(target_dir)
    print(f"[AST] Download complete.")


def download_pann(target_dir: Path) -> None:
    """Download PANN CNN14 weights from Zenodo."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in PANN_FILES.items():
        dest = target_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"[PANN] Downloading {filename}...")
        download_file(url, dest)
    print(f"[PANN] Download complete.")


def download_yamnet(target_dir: Path) -> None:
    """Download YAMNet model from Kaggle."""
    if (target_dir / "saved_model.pb").exists():
        return

    print(f"[YAMNet] Downloading weights to {target_dir}...")
    tmp_cache = target_dir.parent / "_kaggle_tmp"
    os.environ["KAGGLEHUB_CACHE"] = str(tmp_cache)

    import kagglehub

    src = Path(kagglehub.model_download(YAMNET_HANDLE))

    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(src, target_dir)
    shutil.rmtree(tmp_cache, ignore_errors=True)
    print(f"[YAMNet] Download complete.")


def download_all_models(models_dir: Optional[Path] = None) -> bool:
    """Download all required models into models_dir."""
    from ..core.config import DEFAULT_MODELS_DIR
    target = Path(models_dir or DEFAULT_MODELS_DIR).resolve()
    target.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("      AudioSense: Downloading Core Acoustic Models")
    print(f" Target Directory: {target}")
    print("=" * 60)

    try:
        download_ast(target / "ast")
        download_pann(target / "pann")
        download_yamnet(target / "yamnet")
        print("\nAll acoustic models successfully acquired and ready to use!")
        return True
    except Exception as e:
        print(f"\nError downloading models: {e}")
        return False
