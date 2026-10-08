"""
AudioSense Model Downloader.
Downloads pre-trained AST, PANN (CNN14), and YAMNet models into the configured models directory.
Can be run standalone or invoked during package setup.
"""

import os
import shutil
import sys
from pathlib import Path

# Resolve models directory dynamically
try:
    from audiosense.core.config import DEFAULT_MODELS_DIR
    MODELS_DIR = DEFAULT_MODELS_DIR
except ImportError:
    ROOT = Path(__file__).resolve().parent
    MODELS_DIR = ROOT / "models"

AST_DIR = MODELS_DIR / "ast"
PANN_DIR = MODELS_DIR / "pann"
YAMNET_DIR = MODELS_DIR / "yamnet"

AST_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

PANN_FILES = {
    "Cnn14_mAP=0.431.pth":
        "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
    "class_labels_indices.csv":
        "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
}

YAMNET_HANDLE = "google/yamnet/tensorFlow2/yamnet"


def download_file(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Stream a file to disk with a simple progress readout."""
    import requests

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
                    print(f"\r    {dest.name}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB",
                          end="", flush=True)
        print()
    tmp.replace(dest)


def download_ast(target_dir: Path = AST_DIR) -> None:
    if (target_dir / "config.json").exists() and (
        any(target_dir.glob("*.safetensors")) or any(target_dir.glob("*.bin"))
    ):
        print("[AST] already present, skipping")
        return

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    print("[AST] downloading from HuggingFace...")
    target_dir.mkdir(parents=True, exist_ok=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(AST_NAME)
    model = AutoModelForAudioClassification.from_pretrained(AST_NAME)
    feature_extractor.save_pretrained(target_dir)
    model.save_pretrained(target_dir)
    print(f"[AST] saved to {target_dir}")


def download_pann(target_dir: Path = PANN_DIR) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print("[PANN] checking files...")
    for filename, url in PANN_FILES.items():
        dest = target_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"    {filename}: already present, skipping")
            continue
        download_file(url, dest)
    print(f"[PANN] saved to {target_dir}")


def download_yamnet(target_dir: Path = YAMNET_DIR) -> None:
    if (target_dir / "saved_model.pb").exists():
        print("[YAMNet] already present, skipping")
        return

    print("[YAMNet] downloading from Kaggle...")
    tmp_cache = target_dir.parent / "_kaggle_tmp"
    os.environ["KAGGLEHUB_CACHE"] = str(tmp_cache)

    import kagglehub

    src = Path(kagglehub.model_download(YAMNET_HANDLE))

    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(src, target_dir)
    shutil.rmtree(tmp_cache, ignore_errors=True)
    print(f"[YAMNet] saved to {target_dir}")


def download_all_models(base_dir: Path = MODELS_DIR) -> int:
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models directory: {base_dir}\n")

    results = {}
    tasks = [
        ("AST", lambda: download_ast(base_dir / "ast")),
        ("PANN", lambda: download_pann(base_dir / "pann")),
        ("YAMNet", lambda: download_yamnet(base_dir / "yamnet")),
    ]

    for name, fn in tasks:
        try:
            fn()
            results[name] = "OK"
        except Exception as e:
            results[name] = f"FAILED: {type(e).__name__}: {e}"
        print()

    print("=" * 50)
    for name, status in results.items():
        print(f"{name:8s} {status}")
    print("=" * 50)
    return 0 if all(s == "OK" for s in results.values()) else 1


if __name__ == "__main__":
    sys.exit(download_all_models())
