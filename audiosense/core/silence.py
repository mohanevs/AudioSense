"""
AudioSense Global Warning, Log, and Progress Bar Suppression Engine.
Completely suppresses library logs, oneDNN notices, TensorFlow deprecations,
tqdm progress bars, and native C++ STDERR messages.
"""

import os
import sys
import contextlib
import logging
import warnings

# Environment variables (set before any heavy framework imports)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"
os.environ["GLOG_minloglevel"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KAGGLEHUB_VERBOSITY"] = "quiet"

# Python Warnings & Loggers
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

for noisy_logger in [
    "tensorflow",
    "tensorflow_hub",
    "absl",
    "torch",
    "transformers",
    "urllib3",
    "panns_inference",
    "faster_whisper",
]:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
except ImportError:
    pass


# Global TQDM suppression
def disable_tqdm():
    """Silences all tqdm progress bars throughout the Python session."""
    try:
        import tqdm
        orig_init = tqdm.tqdm.__init__

        def _silent_init(self, *args, **kwargs):
            kwargs["disable"] = True
            orig_init(self, *args, **kwargs)

        tqdm.tqdm.__init__ = _silent_init

        import tqdm.auto
        tqdm.auto.tqdm.__init__ = _silent_init
    except Exception:
        pass


disable_tqdm()


# OS and Python Level Stream Redirection
@contextlib.contextmanager
def silence_all():
    """
    Suppresses stdout, stderr, and OS file descriptors (capturing C++ / native logs).
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    c_stdout_saved = None
    c_stderr_saved = None
    devnull_fd = None

    try:
        try:
            devnull_fd = os.open(os.devnull, os.O_RDWR)
            c_stdout_saved = os.dup(1)
            c_stderr_saved = os.dup(2)
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
        except (OSError, AttributeError):
            pass

        with open(os.devnull, "w", encoding="utf-8") as devnull_py:
            sys.stdout = devnull_py
            sys.stderr = devnull_py
            yield

    finally:
        if c_stdout_saved is not None:
            try:
                os.dup2(c_stdout_saved, 1)
                os.close(c_stdout_saved)
            except OSError:
                pass
        if c_stderr_saved is not None:
            try:
                os.dup2(c_stderr_saved, 2)
                os.close(c_stderr_saved)
            except OSError:
                pass
        if devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except OSError:
                pass

        sys.stdout = old_stdout
        sys.stderr = old_stderr
