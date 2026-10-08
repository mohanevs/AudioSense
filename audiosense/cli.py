"""
Command-line interface utilities for AudioSense setup and downloads.
"""

import sys
from .models.download import download_all_models
from .context.download_llm import download_llm_model


def download_cli():
    """CLI command to download core models."""
    print("Running AudioSense core model download...")
    success = download_all_models()
    sys.exit(0 if success else 1)


def setup_cli():
    """CLI command to download both core models and local LLM."""
    print("Running complete AudioSense environment setup...")
    m_ok = download_all_models()
    print()
    try:
        download_llm_model()
        l_ok = True
    except Exception as e:
        print(f"Notice: LLM download encountered an issue: {e}")
        l_ok = False

    sys.exit(0 if m_ok else 1)


def main():
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("setup", "install-models", "download"):
        if "llm" in sys.argv:
            setup_cli()
        else:
            download_cli()
    else:
        print("AudioSense CLI")
        print("Usage:")
        print("  audiosense download      # Download core acoustic models (AST, PANN, YAMNet)")
        print("  audiosense setup --llm   # Download core models + Phi-4-mini LLM")
