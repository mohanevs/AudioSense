"""
AudioSense LLM Setup & Downloader.
Downloads Phi-4-mini GGUF model and checks/installs llama-server.
Can be run standalone or invoked during package setup.
"""

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Resolve models directory dynamically
try:
    from audiosense.core.config import DEFAULT_MODELS_DIR
    MODELS_DIR = DEFAULT_MODELS_DIR
except ImportError:
    ROOT = Path(__file__).resolve().parent
    MODELS_DIR = ROOT / "models"

LLM_DIR = MODELS_DIR / "llm"

MODEL_FILENAME = "Phi-4-mini-instruct-Q4_K_M.gguf"
MODEL_URL = (
    "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF"
    "/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
)
MIN_VALID_SIZE = 1_000_000_000

SERVER_NAME = "llama-server.exe" if os.name == "nt" else "llama-server"


def find_llama_server() -> str | None:
    found = shutil.which(SERVER_NAME)
    if found:
        return found

    candidates = []
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", "")
        winget = os.path.join(local, "Microsoft", "WinGet")
        candidates += glob.glob(os.path.join(winget, "Links", SERVER_NAME))
        candidates += glob.glob(os.path.join(winget, "Packages", "*", "**", SERVER_NAME), recursive=True)
    else:
        candidates += ["/opt/homebrew/bin/llama-server", "/usr/local/bin/llama-server"]

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def install_llama_cpp() -> bool:
    if os.name == "nt":
        if not shutil.which("winget"):
            print("    winget was not found. Install 'App Installer' from the Microsoft Store,")
            print("    or download llama.cpp manually from https://github.com/ggml-org/llama.cpp/releases")
            return False
        cmd = ["winget", "install", "llama.cpp",
               "--accept-package-agreements", "--accept-source-agreements"]
    elif sys.platform == "darwin":
        if not shutil.which("brew"):
            print("    Homebrew was not found. Install it from https://brew.sh then re-run.")
            return False
        cmd = ["brew", "install", "llama.cpp"]
    else:
        print("    Automatic install is only supported on Windows and macOS.")
        print("    On Linux, build or download llama.cpp from https://github.com/ggml-org/llama.cpp")
        return False

    print(f"    Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def ensure_llama_cpp() -> None:
    print("[1/2] Checking llama.cpp...")
    path = find_llama_server()
    if not path:
        print("    llama-server not found. Attempting installation...")
        if not install_llama_cpp():
            print("    Notice: Proceeding without llama-server. Ollama or internal synthesis will be used.")
            return
        path = find_llama_server()
    if path:
        os.environ["PATH"] = os.path.dirname(path) + os.pathsep + os.environ["PATH"]
        print(f"    llama.cpp ready: {path}")


def download_resumable(url: str, dest: Path, chunk: int = 1 << 20, retries: int = 3) -> None:
    import requests
    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        resume_from = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

        try:
            with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()

                if resume_from and r.status_code == 206:
                    mode, done = "ab", resume_from
                    total = resume_from + int(r.headers.get("content-length", 0))
                else:
                    mode, done = "wb", 0
                    total = int(r.headers.get("content-length", 0))

                with open(part, mode) as f:
                    for block in r.iter_content(chunk_size=chunk):
                        f.write(block)
                        done += len(block)
                        if total:
                            print(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB "
                                  f"({done * 100 / total:5.1f}%)", end="", flush=True)
                print()
            break
        except Exception as e:
            print(f"\n    Download interrupted ({type(e).__name__}). Attempt {attempt}/{retries}")
            if attempt == retries:
                raise

    part.replace(dest)


def download_llm_model(target_dir: Path = LLM_DIR) -> Path:
    print("[2/2] Checking Phi-4-mini model...")
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / MODEL_FILENAME

    if dest.exists() and dest.stat().st_size >= MIN_VALID_SIZE:
        print(f"    Already present, skipping: {dest}")
        return dest

    print("    Downloading Phi-4-mini-instruct Q4_K_M (~2.5 GB)...")
    download_resumable(MODEL_URL, dest)

    if dest.stat().st_size < MIN_VALID_SIZE:
        dest.unlink()
        raise RuntimeError("Downloaded file is too small or corrupt. Re-run to retry.")
    print(f"    Saved to {dest}")
    return dest


def download_all_llm(base_dir: Path = MODELS_DIR) -> int:
    print("=" * 50)
    print("      AudioSense - LLM Setup")
    print("=" * 50)

    try:
        ensure_llama_cpp()
        print()
        dest = download_llm_model(base_dir / "llm")
        print("\n" + "=" * 50)
        print("         LLM SETUP COMPLETE")
        print("=" * 50)
        print(f"Model: {dest}")
        return 0
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(download_all_llm())