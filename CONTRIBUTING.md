<div align="center">

# 🤝 Contributing to AudioSense

**Thanks for helping machines hear better!** Every contribution counts — bug reports, docs, new sound labels, model adapters, and tests.

[Report a Bug](https://github.com/mohanevs/AudioSense/issues/new?template=bug_report.yml) •
[Request a Feature](https://github.com/mohanevs/AudioSense/issues/new?template=feature_request.yml) •
[Discussions & Issues](https://github.com/mohanevs/AudioSense/issues)

</div>

---

## 📑 Table of Contents

1. [Code of Conduct](#-code-of-conduct)
2. [Ways to Contribute](#-ways-to-contribute)
3. [Development Setup](#-development-setup)
4. [Project Map](#-project-map)
5. [Workflow: From Idea to Pull Request](#-workflow-from-idea-to-pull-request)
6. [Coding Guidelines](#-coding-guidelines)
7. [Testing Your Changes](#-testing-your-changes)
8. [Commit Messages](#-commit-messages)
9. [Pull Request Checklist](#-pull-request-checklist)
10. [Reporting Bugs & Requesting Features](#-reporting-bugs--requesting-features)
11. [License](#-license)

---

## 📜 Code of Conduct

Be respectful, patient and constructive. We welcome contributors of every background and experience level. Harassment, personal attacks and discrimination are not tolerated. Maintainers may edit or remove comments, commits and issues that violate this standard.

---

## 💡 Ways to Contribute

You do **not** need to be an audio or ML expert to help.

| Type | Examples |
|---|---|
| 🐛 **Bug fixes** | Wrong predictions routing, crashes on odd audio formats, Windows/macOS path problems |
| ✨ **Features** | New model adapters, new LLM backends, extra output languages, live-loop improvements |
| 🔊 **Labels & routing** | Improve speech vs. environment indicators in the Switcher, taxonomy fixes |
| 📚 **Documentation** | Clearer README, new examples, docstrings, tutorials |
| 🧪 **Tests** | The project currently has no automated test suite — adding `pytest` tests is highly valued |
| 🌍 **Translations** | Better multilingual output in `context/translator.py` |
| 💬 **Support** | Answer questions and triage issues |

New here? Look for issues labelled **`good first issue`** or **`help wanted`**.

---

## 🛠️ Development Setup

**Requirements:** Python **3.10 – 3.12**, `git`, and ~3 GB free disk space (more if you download the optional local LLM).

### 1. Fork & clone

```bash
# Fork the repo on GitHub first, then:
git clone https://github.com/<your-username>/AudioSense.git
cd AudioSense
git remote add upstream https://github.com/mohanevs/AudioSense.git
```

### 2. Create a virtual environment

```bash
python -m venv env

# Windows
.\env\Scripts\activate
# Linux / macOS
source env/bin/activate
```

### 3. Install in editable mode

```bash
pip install -e .
# or, to mirror the README setup exactly:
pip install -r requirements.txt
```

### 4. Download model weights

```bash
python load_model.py        # AST, PANN (CNN14), YAMNet
python load_llm.py          # Optional: local Phi-4-mini (~2.5 GB)
```

You can also use the CLI: `audiosense download` or `audiosense setup --llm`.

> ⚠️ Model weights are large and are **git-ignored** (`models/*`). Never commit weights, `.wav` / `.mp3` / `.flac` files, or generated files such as `audio_sense_labels.json`.

### 5. Smoke test

```python
from audiosense import AudioSense

model = AudioSense(device="auto", language="en")
print(model.predict("path/to/your_audio.wav"))
```

---

## 🗺️ Project Map

Know where to look before you change something:

| Area | Path | What lives here |
|---|---|---|
| Engine | `audiosense/core/` | `engine.py` (main `AudioSense` class), `config.py`, `silence.py` |
| Audio I/O | `audiosense/audio/` | Format conversion, 16 kHz / 32 kHz preprocessing |
| Models | `audiosense/models/` | AST, PANN, YAMNet, Whisper adapters + lazy `ModelManager` |
| Labels | `audiosense/label.py`, `audiosense/labels/` | Functional label APIs, AudioSet taxonomy |
| Switcher | `audiosense/switcher/` | Speech-vs-environment router and Silero VAD |
| Context / LLM | `audiosense/context/` | LLM client, translator, context & DB summaries |
| Storage | `audiosense/storage/` | JSON event database (`view_db`) |
| Live mic | `audiosense/live/` | Real-time listening loop |
| CLI | `audiosense/cli.py` | `audiosense`, `audiosense-download`, `audiosense-setup` |

### Common contribution recipes

<details>
<summary><b>➕ Adding a new model adapter</b></summary>

1. Create `audiosense/models/<name>_model.py` with a class exposing the same prediction interface as the existing adapters (see `ast_model.py` or `yamnet_model.py`).
2. Register it as a **lazy property** in `ModelManager` (`models/manager.py`) so it only loads when first used.
3. Add its download logic to `models/download.py` if it needs weights.
4. Wire it into the ensemble in `core/engine.py` and document the weight/consensus behaviour.
5. Update the README feature table and project structure.
</details>

<details>
<summary><b>🔀 Tuning speech / environment routing</b></summary>

Routing indicators live in `audiosense/switcher/router.py` (`DEFAULT_SPEECH_INDICATORS`, `DEFAULT_NON_SPEECH_HUMAN_SOUNDS`). When you add or move a label, explain with an example audio description *why* it belongs there.
</details>

<details>
<summary><b>🌐 Adding an LLM backend</b></summary>

Extend `audiosense/context/llm.py`. Keep backends optional — the core package must keep working without a local LLM installed.
</details>

---

## 🔄 Workflow: From Idea to Pull Request

1. **Search first** — check existing [issues](https://github.com/mohanevs/AudioSense/issues) and PRs to avoid duplicates.
2. **Open an issue** for anything non-trivial (new feature, API change, large refactor) and wait for maintainer feedback before investing lots of time. Small fixes and doc typos can go straight to a PR.
3. **Sync your fork**
```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
```
4. **Create a branch** with a descriptive name:
```bash
   git checkout -b feat/live-loop-vad-threshold
```
   Prefixes: `feat/`, `fix/`, `docs/`, `test/`, `refactor/`, `chore/`.
5. **Make focused changes** — one logical change per PR.
6. **Test** your change (see [below](#-testing-your-changes)).
7. **Push and open a Pull Request** against `main`, filling in the PR template.
8. **Respond to review** — push follow-up commits to the same branch; the PR updates automatically.

---

## 🧑‍💻 Coding Guidelines

- **Style:** follow [PEP 8](https://peps.python.org/pep-0008/); 4-space indentation; keep lines reasonably short (≈100 chars).
- **Type hints** on public functions and methods, matching the existing code (`Optional[...]`, `Union[...]`).
- **Docstrings** on all public classes and functions explaining arguments and return values.
- **Public API stays stable.** Anything exported in `audiosense/__init__.py` is public. Avoid breaking changes; if unavoidable, call them out clearly in your PR.
- **Lazy loading.** Heavy models (torch / TensorFlow / Whisper) must load on first use, never at import time.
- **Keep logs quiet.** AudioSense deliberately silences TensorFlow/oneDNN/tqdm noise (`core/silence.py`). Don't add stray `print` calls or re-enable noisy logging in library code. (CLI scripts may print user-facing messages.)
- **No hard-coded paths.** Use `core/config.py` helpers and `pathlib.Path` — the library supports Linux, macOS and Windows.
- **Optional dependencies stay optional.** Don't force heavy new dependencies on all users; if you must add one, justify it in the PR and update **both** `pyproject.toml` and `requirements.txt`.
- **Don't commit large or private files** (weights, audio recordings, personal paths, API keys).

---

## 🧪 Testing Your Changes

There is no formal test suite yet, so please verify manually and, where you can, add tests.

**Manual checks (minimum):**

```python
from audiosense import AudioSense, convert_to_wav

model = AudioSense(device="auto", language="en")

# Environmental sound -> expect route == "context"
print(model.predict("samples/environment.wav"))

# Human speech -> expect route == "transcription"
print(model.predict("samples/speech.wav"))
```

Also check the areas your change touches, for example:

- file conversion with a non-WAV input (`.mp3`, `.m4a`, …)
- `model.translation("hin")` if you changed translation
- `model.loop(chunk_duration=2.0)` if you changed the live loop
- `import audiosense` still works **without** weights downloaded

**Adding automated tests (encouraged):** create a `tests/` folder and use [`pytest`](https://docs.pytest.org/):

```bash
pip install pytest
pytest -q
```

Prefer small, synthetic audio (e.g. generated sine waves / silence with `numpy`) over large audio files, and mock model inference where possible so tests run fast and without downloading weights.

> `test.py` in the repo root is a personal smoke-test script with a local file path — don't rely on it as a test; feel free to replace it with real tests.

---

## ✍️ Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary in imperative mood>

<optional body explaining what and why>
```

**Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`.

Examples:

```
feat(switcher): allow runtime override of speech indicators
fix(audio): handle mono 8 kHz input in preprocess
docs(readme): add Ollama endpoint example
```

---

## ✅ Pull Request Checklist

Before you click **Create pull request**, confirm:

- [ ] The PR targets `main` and is based on the latest upstream `main`
- [ ] It does **one** thing and has a clear title and description
- [ ] Code follows the guidelines above, with type hints and docstrings
- [ ] I tested the change locally (and added tests if practical)
- [ ] No weights, audio files, secrets or personal paths are committed
- [ ] README / docstrings are updated if behaviour or API changed
- [ ] New dependencies (if any) are added to both `pyproject.toml` and `requirements.txt`

Maintainers review PRs as time allows. Please be patient — a friendly nudge after a week is fine.

---

## 🐞 Reporting Bugs & Requesting Features

**Bugs** — use the [bug report form](https://github.com/mohanevs/AudioSense/issues/new?template=bug_report.yml) and include:

- OS, Python version and AudioSense version (`audiosense.__version__`)
- Whether you use CPU or GPU (`device=`)
- A minimal code snippet that reproduces the problem
- The full error traceback / unexpected output
- Audio format and rough duration (don't upload private recordings)

**Features** — use the [feature request form](https://github.com/mohanevs/AudioSense/issues/new?template=feature_request.yml) and describe the problem you're solving, not only the solution you have in mind.

**Security issues** — please do **not** open a public issue. Contact the maintainer privately via the email listed on the maintainer's GitHub profile.

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

<div align="center">

**Happy hacking — and thank you for making AudioSense better! 🎙️**

</div>
