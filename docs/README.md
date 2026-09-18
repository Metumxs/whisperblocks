# 🎙️ WhisperBlocks
> Local transcription pipeline — from audio to LLM-ready text blocks.

> High-performance, fully local batch audio/video transcription using [`WhisperX`](https://github.com/m-bain/whisperX), optimized for NVIDIA GPUs (CUDA).
> Designed for transcribing Ukrainian lectures, meetings and raw audio into structured text or subtitle files — **no cloud, no API keys, no data leaves your machine during transcription** (except for optional Hugging Face model downloads during setup).

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#-license)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-0078D6.svg)](#-requirements)

---

## 📑 Table of Contents

- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Output Files](#-output-files)
- [Speaker Diarization](#-speaker-diarization)
- [Auto-Preprocessing](#-auto-preprocessing)
- [LLM Post-Processing](#-llm-post-processing)
- [Model Selection](#-model-selection)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq)
- [License](#-license)

---

## ✨ Features

- **100% local transcription** — audio never leaves your machine after the models are downloaded once.
- **GPU accelerated** — CUDA 12 + cuDNN via pip wheels, no manual CUDA Toolkit install.
- **Batched decoding** — WhisperX batches segments for large speedups on long files.
- **Word-level alignment** — forced alignment against a language-specific phoneme model produces millisecond-accurate word timings.
- **Optional speaker diarization** — pyannote.audio v4 tags every segment with a speaker label; runs as a third pipeline stage.
- **Batch processing** — drop a list of files into `config.toml` and walk away.
- **Four output modes** — timestamped text, clean plain text, `.srt` subtitles, or all at once.
- **Filler removal** — hints the model to drop verbal tics (`ну`, `типу`, `от`, long pauses) during transcription.
- **Auto-preprocessing** — raw timestamped output is automatically merged into fixed-duration blocks (60 s default) for downstream AI processing.
- **Ready-made LLM prompts** — bundled system prompts (Ukrainian + English) for cleaning up transcripts without paraphrasing.
- **Model-aware filenames** — every output file carries the model name (`--large-v3-turbo`) so you can compare runs.
- **Isolated environment** — dependencies live in a project-local `.venv`, never in your system Python.
- **Secret-safe configuration** — your `config.toml` (which holds your Hugging Face token) is gitignored; a tracked template seeds it on first run.
- **One-click setup + launcher** — `install.bat` for the first run, `run.bat` afterwards.

---

## 🖥️ Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10 (21H2) | Windows 11 |
| **GPU** | NVIDIA, 6 GB VRAM | NVIDIA, 8 GB+ VRAM (RTX 3060 / 4060 or better) |
| **Python** | 3.12 (pinned; WhisperX 3.8.6 requires `~=2.8.0` for torch) | **3.12** |
| **Disk** | ~8 GB free (PyTorch + models + runtimes) | SSD with ~20 GB free |
| **CPU-only fallback** | any modern x86-64 | — (works, but 10–20× slower) |

> **No NVIDIA GPU?** Set `device = "cpu"` in `config.toml` and use a smaller model such as `small` or `medium`. Note that `install.bat` still installs the CUDA build of PyTorch — the CPU build is not supported by the current dependency chain.

> **Why Python 3.12 specifically?** WhisperX 3.8.6 pins `Requires-Python >=3.10,<3.14` and its CUDA build of `torch` only ships wheels up to 3.12. Python 3.12 is the newest version where the whole stack is known to work.

---

## 📦 Installation

WhisperBlocks runs inside a **project-local virtual environment** (`.venv`). The setup is a two-step process: **install once, run many times**.

### Step 1 — One-time setup (`install.bat`)

Double-click **`install.bat`** in the project folder, or run it from a terminal:

```bash
install.bat
```

The script does the following, in order:

1. Verifies Python 3.12 is available via the `py` launcher.
2. Creates `.venv\` (reuses it if it already exists).
3. Upgrades `pip` inside the venv.
4. **Installs CUDA-enabled PyTorch from the PyTorch index** (`cu128`) — this is the critical step.
5. Installs WhisperX and the remaining dependencies from `requirements.txt`.
6. Verifies `torch.cuda.is_available()` returns `True`.

If Python 3.12 is not installed yet, the script prints the exact command to install it (`py install 3.12`). Do that, then re-run `install.bat`.

### Step 2 — Run (`run.bat`)

Double-click **`run.bat`**. It activates `.venv` and starts `src/transcript.py`.

- If `.venv` is missing, `run.bat` exits with a message telling you to run `install.bat` first.
- If `config/config.toml` is missing (first run ever), `transcript.py` copies `config/config.toml.example` into place, prints a hint panel, and exits so you can fill in your file list before the first real run.

### Manual setup (if you prefer doing it by hand)

The commands `install.bat` runs are:

```bash
# 1. Create the virtual environment with Python 3.12
py -3.12 -m venv .venv

# 2. Activate it
.venv\Scripts\activate

# 3. Upgrade pip (use the "python -m pip" form on Windows)
python -m pip install --upgrade pip

# 4. Install CUDA-enabled PyTorch FIRST
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 5. Install everything else
pip install -r requirements.txt
```

> **Do not skip step 4.** PyPI ships the CPU-only PyTorch build on Windows. Running `pip install -r requirements.txt` alone produces an environment where WhisperX loads fine but crashes at the first GPU call with `Torch not compiled with CUDA enabled`.

A condensed cheat-sheet of these commands (with recovery steps and a list of safe-to-ignore warnings) ships as `SETUP-NOTES.txt` in the project root.

### What gets installed

| Package | Why |
|---------|-----|
| `whisperx==3.8.6` | Transcription + alignment + diarization pipeline |
| `rich` | Terminal dashboard, live segment output, panels |
| `torch==2.8.0` + `torchaudio==2.8.0` | Neural network runtime (CUDA build from the PyTorch index) |
| `nvidia-cublas-cu12` | cuBLAS runtime DLLs (Windows; avoids a full CUDA Toolkit install) |
| `nvidia-cudnn-cu12` | cuDNN runtime DLLs (Windows) |

> `transcript.py` automatically injects the CUDA DLL paths from these wheels into the process at startup — you do **not** need to install the NVIDIA CUDA Toolkit separately.

### Optional: install FFmpeg

WhisperX works without it, but pyannote will print a `torchcodec` warning every run and fall back to a slower audio decoder. Installing FFmpeg system-wide removes the warning and speeds up audio loading:

```bash
winget install Gyan.FFmpeg
```

Close and reopen the terminal afterwards so the new `PATH` takes effect.

### Changing the CUDA version

The setup script targets `cu128` (CUDA 12.8), which works on RTX 30-series and 40-series cards with recent drivers. If your NVIDIA driver is older and `cu128` fails, edit `install.bat` and swap the index URL:

```batch
--index-url https://download.pytorch.org/whl/cu126
```

A list of all supported combinations lives at <https://download.pytorch.org/whl/>. Torch version must stay at `2.8.0` — WhisperX 3.8.6 pins `torch~=2.8.0`.

---

## 🗂️ Project Structure

```text
TRANSCRIBE-WhisperBlocks/
├── install.bat                ← One-time setup (creates .venv, installs deps)
├── run.bat                    ← One-click launcher (activates .venv)
├── requirements.txt           ← Python dependencies
├── SETUP-NOTES.txt            ← Cheat-sheet for manual / recovery installs
├── .gitignore                 ← Excludes .venv/, config.toml, media, outputs
├── .venv/                     ← Virtual environment (created by install.bat)
├── config/
│   ├── config.toml            ← YOUR config (gitignored; created on first run)
│   └── config.toml.example    ← Template (tracked in git)
├── src/
│   ├── transcript.py          ← Main WhisperX pipeline (ASR → align → diarize)
│   └── preprocess.py          ← Merges raw segments into fixed-duration blocks
├── prompts/
│   ├── UK-llm-ai-processing-prompt.txt    ← Ukrainian cleanup prompt
│   └── EN-llm-ai-processing-prompt.txt    ← English cleanup prompt
├── docs/
│   ├── README.md              ← This file
│   └── LICENSE                ← MIT license text
│
├── lecture.mp3                ← Your audio files live here (project root)
└── ...                        ← Output files land here too
```

**Key rule:** audio files and generated outputs live in the **project root** (next to `run.bat`). The `config/` and `src/` folders are for tooling only.

---

## ⚙️ Configuration

### How `config.toml` works

WhisperBlocks keeps **two** config files in `config/`:

| File | Tracked in git? | Who edits it |
|------|-----------------|--------------|
| `config.toml` | ❌ No (gitignored) | You — this is your live config |
| `config.toml.example` | ✅ Yes | Nobody — it is a template |

`config.toml` is gitignored because it holds your Hugging Face access token (`hf_token`) whenever speaker diarization is enabled. A token pasted into a public repository is a leaked token — Hugging Face will revoke it, but the exposure window still counts.

**On first run** (when `config/config.toml` does not exist), `transcript.py` copies `config.toml.example` into place, prints a hint panel, and exits. You then edit `config.toml` to add your file list and re-run `run.bat`. This guarantees you never accidentally push your personal configuration.

**Never edit `config.toml.example`.** Your changes to it will be ignored — it only exists to seed `config.toml` on first run.

> **Already committed `config.toml` before this pattern existed?** Adding it to `.gitignore` does not remove it from git history. Run `git rm --cached config/config.toml` and commit, then rotate your Hugging Face token immediately.

### Editing `config.toml`

```toml
# ── Files to transcribe ─────────────────────────────────────────────
files = [
    "lecture.mp4",
    "meeting.wav",
    "C:/Audio/interview.m4a"
]

# ── Output mode ─────────────────────────────────────────────────────
output_mode = "all"

# ── Filler removal ──────────────────────────────────────────────────
remove_fillers = true

# ── Auto-preprocessing ──────────────────────────────────────────────
auto_preprocess_block_seconds = 60

# ── Diarization ─────────────────────────────────────────────────────
diarize = false
hf_token = ""
min_speakers = 0
max_speakers = 0

# ── Model & hardware ────────────────────────────────────────────────
model_size = "large-v3-turbo"
device     = "cuda"
language   = "uk"
```

### Option reference

| Key | Type | Accepted values | Default |
|-----|------|-----------------|---------|
| `files` | array | Any relative/absolute media path | `[]` |
| `output_mode` | string | `timecode` \| `simple` \| `srt` \| `all` | `all` |
| `remove_fillers` | bool | `true` \| `false` | `true` |
| `auto_preprocess_block_seconds` | number | `0` (off) or any positive number | `60` |
| `diarize` | bool | `true` \| `false` | `false` |
| `hf_token` | string | `hf_...` token from Hugging Face | `""` |
| `min_speakers` | number | `0` (auto) or any positive integer | `0` |
| `max_speakers` | number | `0` (auto) or any positive integer | `0` |
| `model_size` | string | See [Model Selection](#-model-selection) | `large-v3-turbo` |
| `device` | string | `cuda` \| `cpu` | `cuda` |
| `language` | string | ISO-639-1 code, or `auto` | `uk` |

> **On `language = "auto"`:** Whisper's language detector is unreliable for Ukrainian — it frequently classifies short or noisy segments as Russian. Use `"auto"` only for genuinely mixed-language batches. For Ukrainian-only audio, always set `"uk"` explicitly.

---

## ▶️ Usage

1. Drop your audio/video files into the **project root** (the folder with `run.bat`).
2. List them in `config/config.toml` under `files = [...]`.
3. Double-click **`run.bat`** — it activates `.venv` and runs the pipeline.

   Or, from a terminal with `.venv` already active:

   ```bash
   python src/transcript.py
   ```

4. Watch the live console output. When done, output files appear next to the source audio.

---

## 📄 Output Files

Every output carries a **prefix** (mode) and a **suffix** (model name). Example for `zvvb_pz1_meet.wav` transcribed with `large-v3-turbo`:

| File | Description |
|------|-------------|
| `timecode-zvvb_pz1_meet--large-v3-turbo.txt` | Full transcript with `[MM:SS -> MM:SS]` timestamps on every segment |
| `timecode-zvvb_pz1_meet--large-v3-turbo.blocks.txt` | The above, merged into ~60-second blocks (auto-generated) |
| `simple-zvvb_pz1_meet--large-v3-turbo.txt` | Plain text, one segment per line, no timestamps |
| `zvvb_pz1_meet--large-v3-turbo.srt` | Standard `.srt` subtitle file |

When diarization is enabled, every line in the `.txt` and `.srt` files is prefixed with `[SPEAKER_NN]`.

**Why the model name in the filename?** So you can transcribe the same audio with `medium` and `large-v3-turbo` and compare results side by side without renaming anything.

### Format examples

**`timecode`** — segment-by-segment:

```text
[00:00 -> 00:05] Добрий день, сьогодні ми розглянемо...
[00:05 -> 00:11] ...основи машинного навчання.
```

**`simple`** — plain text for pasting into an LLM:

```text
Добрий день, сьогодні ми розглянемо основи машинного навчання.
```

**`.srt`** — standard subtitles for VLC, YouTube, Premiere:

```srt
1
00:00:00,000 --> 00:00:05,000
Добрий день, сьогодні ми розглянемо...

2
00:00:05,000 --> 00:00:11,000
...основи машинного навчання.
```

**With diarization enabled** (`diarize = true`), every line gains a speaker tag:

```text
[00:00 -> 00:05] [SPEAKER_00] Добрий день, сьогодні ми розглянемо...
[00:05 -> 00:11] [SPEAKER_01] ...основи машинного навчання.
```

---

## 🎤 Speaker Diarization

Diarization answers **"who said what"** — it partitions the audio into speaker turns and labels them `SPEAKER_00`, `SPEAKER_01`, etc. It is powered by [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) and runs as the third WhisperX pipeline stage.

> **WhisperX 3.8.x uses pyannote-audio v4.** The default diarization model is `pyannote/speaker-diarization-community-1` (CC-BY-4.0). Older WhisperX releases used the pair `speaker-diarization-3.1` + `segmentation-3.0` — those are no longer the default and are not required.

### Why the license gate?

Pyannote's diarization models are **gated** on Hugging Face: you cannot download them without (a) a Hugging Face account, (b) accepting their license agreements, and (c) an access token. This is a legal gate, not a technical one, but it is enforced by the model host. The `community-1` model is auto-approved — there is no waiting list, but you still have to click through the agreement once.

### One-time setup

1. **Log in** at <https://huggingface.co>.
2. **Accept the license agreement** for the diarization model:
   - <https://huggingface.co/pyannote/speaker-diarization-community-1>
   The page has an **"Agree and access repository"** button at the top. Without clicking it, model downloads will return `401 Unauthorized`.
3. **Create an access token** at <https://huggingface.co/settings/tokens> — a "read"-scoped token is enough.
4. Paste the token into `config/config.toml` under `hf_token = "hf_..."` and set `diarize = true`.

> **Using an older WhisperX (< 3.8)?** That version used pyannote-audio 3.x, whose models are `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`. Accept both of those agreements instead. The current `requirements.txt` pins WhisperX 3.8.6, so you only need the `community-1` agreement.

### Speaker-count hints

If the number of speakers in the audio is known ahead of time, telling the model speeds up diarization and improves accuracy:

- **Unknown count** — leave `min_speakers = 0` and `max_speakers = 0`. The model decides.
- **Exact count** — set both to the same value (e.g. 4 comedians on stage → `min_speakers = 4`, `max_speakers = 4`).
- **Range** — set `min_speakers = 2` and `max_speakers = 5` to bound the search space.

> **Known limitation.** The open-source `community-1` model — the best freely available diarization model as of 2026 — struggles with fast, overlapping dialogue between speakers with similar voices (interviews, panel shows, comedy). Short test clips make this worse: the model needs several minutes of audio to build stable speaker clusters. For a clip that is only 1–2 minutes long, forcing the exact speaker count via `min_speakers` / `max_speakers` is the single most effective tweak.

### VRAM behaviour

Diarization loads a **third** model (~100 MB, but each stage's peak VRAM is what matters). On an 8 GB card, WhisperX cannot keep the ASR, alignment and diarization models resident simultaneously. `transcript.py` explicitly releases each model (`del model; gc.collect(); torch.cuda.empty_cache()`) before loading the next — do not remove that cleanup if you edit the pipeline.

### Trade-offs

| Pros | Cons |
|------|------|
| Segments are labelled with speakers | Adds ~20–40% to runtime per file |
| SRT subtitles get speaker prefixes | Requires a Hugging Face account and license acceptance |
| Useful for interviews / meetings / panel discussions | First run downloads ~100 MB extra |
| | Speaker labels are not stable across different files (SPEAKER_00 in file A is unrelated to SPEAKER_00 in file B) |
| | Quality is imperfect on overlapping speech — see limitation above |

**Diarization is disabled by default.** Turn it on only when you actually need to know who spoke.

---

## 🧩 Auto-Preprocessing

Whisper segments audio by **silence**, not by **meaning**. A sentence can be split across two, three, or five segments, and the resulting transcript is fragmented.

`preprocess.py` solves this by merging consecutive segments into **fixed-duration blocks** (~60 s by default). The result is a much cleaner structure for downstream AI processing.

**Triggered automatically** at the end of every transcription run when `auto_preprocess_block_seconds > 0`.

### Output format

```text
Source: timecode-zvvb_pz1_meet--large-v3-turbo.txt
Duration: 30:46 | Language: uk (100.0%)
Segments: 352 | Blocks: 32 (~60s each)

=== BLOCK 01 | 00:02 → 01:02 ===
Merged text of the block, no timestamps inside.

=== BLOCK 02 | 01:02 → 02:01 ===
...
```

### Manual use

The script also works standalone:

```bash
python src/preprocess.py timecode-*.txt
python src/preprocess.py timecode-*.txt --block 90
python src/preprocess.py timecode-*.txt --outdir blocks/
```

---

## 🤖 LLM Post-Processing

Even after auto-preprocessing, the `.blocks.txt` file is still **raw Whisper output**: full of filler words, false starts, broken punctuation, and occasional ASR errors. Cleaning this up properly requires an LLM.

### Why not automatic?

There is **no API integration** in this tool. The workflow is manual: you copy blocks from the `.blocks.txt` file into your favorite LLM chat (ChatGPT, Claude, local model), and get back cleaned text. This is intentional — it keeps transcription fully offline, and lets you pick whichever LLM you trust.

### Bundled prompts

Two ready-made **system prompts** are included in `prompts/`:

| File | Language | Purpose |
|------|----------|---------|
| `UK-llm-ai-processing-prompt.txt` | Ukrainian | Cleanup rules, prohibitions, and domain dictionary for Ukrainian lectures |
| `EN-llm-ai-processing-prompt.txt` | English | Same for English transcripts |

Both prompts enforce strict rules:

- **No paraphrasing** — the lecturer's exact phrasing is preserved.
- **No shortening** — long sentences stay long.
- **No additions** — no invented headers, summaries, or explanations.
- **Filler removal** — `ну`, `фактично`, `типу`, `так?` etc. dropped only when they act as pause-fillers.
- **False start fixing** — `"я вже, я вже говорила"` → `"я вже говорила"`.
- **Broken sentence closing** — sentences split between blocks are moved back together.
- **ASR error correction** — only when context unambiguously dictates the fix; otherwise the original is kept with a `<!-- ? -->` marker.
- **Grammar & punctuation** — agreement, articles, comma splices fixed without touching meaning.

Each prompt ends with a domain dictionary for business/entrepreneurship lectures — extend it with your own terminology when needed.

### Recommended workflow

1. **Open a new chat** with your LLM.
2. **Paste the system prompt** (`UK-llm-ai-processing-prompt.txt` or `EN-...`) as the first message — or into the dedicated "System" field if the chat has one.
3. **Copy 5–10 blocks** from the `.blocks.txt` file (not all 32 — the LLM loses focus on long inputs).
4. **Ask the LLM** to clean them and return the same `=== BLOCK NN ===` structure.
5. **Repeat** in batches for the remaining blocks.
6. **Concatenate** the cleaned batches back into one file.

### Why batches of 5–10 blocks?

| Reason | Explanation |
|--------|-------------|
| **Focus** | On a 30-minute file with 32 blocks, the model forgets the rules by the middle. |
| **Context** | 5–10 blocks ≈ 7–10 minutes of audio — enough for the model to see cross-block sentence continuations. |
| **Verification** | Easy to spot-check one batch against the original audio. |
| **Recovery** | If the model starts paraphrasing on batch 3, you only lose one batch. |

### Quality tips

- **Verify the first 2–3 blocks manually.** If the LLM is paraphrasing, add to the system prompt: *"Forbidden to reorder words within a sentence."*
- **Chunk the request too, not just the input.** Don't ask "clean 10 blocks and reformat as a summary" — one task per request.
- **Watch for `<-- ? -->` markers.** They indicate the model wasn't confident — go back to the audio and decide manually.
- **Extend the domain dictionary.** Every new lecture introduces new jargon. Add it to the `DOMAIN DICTIONARY` section of the prompt before the next run.

### What the LLM step does **not** do

- It does not run automatically. You paste, it cleans, you copy back.
- It does not touch the audio. Correction is text-only.
- It does not verify against the source. If the LLM hallucinates, you won't catch it — hence the strict prohibitions and the small batch size.

---

## 🧠 Model Selection

| Model | Params | Disk | Speed | Quality |
|-------|--------|------|-------|---------|
| `tiny` | 39 M | ~75 MB | ⚡⚡⚡⚡⚡ | ★☆☆☆☆ |
| `base` | 74 M | ~145 MB | ⚡⚡⚡⚡⚡ | ★★☆☆☆ |
| `small` | 244 M | ~480 MB | ⚡⚡⚡⚡ | ★★★☆☆ |
| `medium` | 769 M | ~1.5 GB | ⚡⚡⚡ | ★★★★☆ |
| `large-v3` | 1.55 B | ~3 GB | ⚡⚡ | ★★★★★ |
| **`large-v3-turbo`** | 809 M | ~1.5 GB | ⚡⚡⚡⚡ | ★★★★★ |

> **For Ukrainian**, `large-v3-turbo` is the sweet spot — near-`large-v3` accuracy at roughly 4× the speed, comfortably fitting an 8 GB card.

**Trade-off summary:** `turbo` occasionally makes small ASR errors on domain jargon (business terminology, proper nouns). `large-v3` is more accurate but ~4× slower. For lectures with heavy terminology, consider `large-v3` if you have the time.

---

## 🔧 Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `[ERROR] Python virtual environment not found` (from `run.bat`) | `.venv` was never created | Run `install.bat` once |
| `[ERROR] Python 3.12 not found` (from `install.bat`) | Python 3.12 is not installed | `py install 3.12` or `winget install Python.Python.3.12`, then re-run `install.bat` |
| `First run — configuration created` panel appears, then the script exits | `config/config.toml` did not exist (expected on the very first run) | Edit `config/config.toml` (file list, language, hf_token) and re-run `run.bat` |
| `no template to copy from at ...config.toml.example` | Archive extracted incompletely | Re-extract the project — `config/config.toml.example` must exist |
| `✕ Failed: ... Torch not compiled with CUDA enabled` | PyTorch CPU build was installed instead of the CUDA build | Reinstall: `pip uninstall -y torch torchaudio` then `pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128` |
| `pip check` reports conflicts involving `torch` version | Wrong CUDA index used (cu124 caps at torch 2.6.0) | Use `cu128` (or `cu126`) — WhisperX 3.8.6 needs `torch~=2.8.0` |
| `module 'whisperx' has no attribute 'DiarizationPipeline'` | WhisperX 3.3.4+ moved the class out of the top-level namespace | Already fixed in `src/transcript.py` — it imports from `whisperx.diarize`. If you see this, your `transcript.py` is out of date. |
| `ModuleNotFoundError: No module named 'whisperx'` | Install ran outside the venv | `.venv\Scripts\activate` first, then `pip install -r requirements.txt` |
| `RuntimeError: Library cublas64_12.dll is not found` | NVIDIA runtime wheels missing | `.venv\Scripts\activate` then `pip install -r requirements.txt` |
| `RuntimeError: Library cudnn_ops_infer64_8.dll is not found` | Same as above | Same as above |
| `UserWarning: torchcodec is not installed correctly` | FFmpeg not on `PATH` | Benign — pyannote falls back to torchaudio. To silence: `winget install Gyan.FFmpeg` |
| `ReproducibilityWarning: TensorFloat-32 (TF32) has been disabled` | pyannote disables TF32 for reproducibility | Benign — informational only |
| `Lightning automatically upgraded your loaded checkpoint` | One-time checkpoint migration | Benign — informational only |
| `huggingface_hub cache uses symlinks ... machine does not support them` | Windows symlink policy | Benign — HF cache works without symlinks. Enable Developer Mode to silence |
| `Xet Storage is enabled ... hf_xet package is not installed` | Optional HF accelerator missing | Benign. Install with `pip install hf_xet` in the venv to speed up downloads |
| `std(): degrees of freedom is <= 0` (pyannote pooling) | Very short segments in the input | Benign — affects short clips more. Use longer audio or set `min_speakers`/`max_speakers` |
| Diarization assigns the wrong speaker to overlapping lines | Known limitation of `community-1` on fast, overlapping dialogue | Try `min_speakers`/`max_speakers` hints, use longer audio, or accept that some lines will be misattributed |
| `CUDA out of memory` at the alignment or diarization stage | Previous model wasn't released | Confirm the `del model` / `flush_vram()` block is still present in `src/transcript.py` |
| `CUDA out of memory` at the ASR stage | Model too large for VRAM | Use `large-v3-turbo` or `medium`, or set `device = "cpu"` |
| `401 Client Error` when diarizing | pyannote license not accepted, or token invalid | Accept the license at `pyannote/speaker-diarization-community-1`, regenerate the HF token, update `config.toml` |
| `Diarization failed` warning, script continues | Any diarization error | The pipeline falls through gracefully — transcript is written without speaker tags |
| `alignment skipped` warning, script continues | Language has no phoneme model on HF | Expected for rare languages; segments keep their ASR timestamps |
| `preprocess.py not importable` | Missing file in `src/` | Ensure `src/preprocess.py` exists next to `src/transcript.py` |
| `✕ Skipped: file not found` | Audio not in project root | Relative paths in `files` are resolved from the project root |
| Garbled or mixed-language output | Wrong `language` value | Set `language = "uk"` explicitly instead of `"auto"` |
| First run hangs for several minutes | Model download in progress | Expected — WhisperX + aligner + (optionally) diarization models are fetched once to `~/.cache/huggingface/hub` |

---

## ❓ FAQ

**Does it work offline?**
Yes, after the first run downloads the model weights. Diarization requires network access only for the very first model download; after that it runs entirely offline.

**Where are the models stored?**
`~/.cache/huggingface/hub` (`C:\Users\<You>\.cache\huggingface\hub` on Windows). This is *outside* the project, so deleting `.venv` does not re-download the model.

**Can I transcribe multiple files at once?**
Yes. List every file in `files = [...]`; they are processed sequentially.

**Does it work on AMD or Intel GPUs?**
Not with CUDA. Set `device = "cpu"` — it works, but expect a large slowdown. The current `install.bat` installs the CUDA build of PyTorch, which still runs on CPU, so no reinstallation is needed to fall back.

**Which video containers are supported?**
Anything `ffmpeg` can decode — `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg` and more.

**Why is the model name in the output filename?**
So you can run the same audio through different models and compare the results directly.

**Can I skip the auto-preprocessing step?**
Yes. Set `auto_preprocess_block_seconds = 0` in the config, or run `python src/preprocess.py` manually later.

**Can I use a different LLM for cleanup?**
Yes — the prompts are plain text, model-agnostic. They work with any chat LLM. For fully offline processing, use a local model via Ollama or LM Studio with a 30B+ parameter model.

**Why a virtual environment instead of a system-wide install?**
WhisperX pins specific versions of `torch`, `ctranslate2` and `pyannote` that will collide with other Python projects on the same machine. `.venv` isolates them and makes the environment disposable — deleting `.venv\` has zero effect on your system Python.

**Where is the virtual environment, and can I delete it?**
It lives at `.venv\` inside the project root. Yes, you can delete it at any time — recreate it by re-running `install.bat`. It is listed in `.gitignore` and never committed.

**Why isn't `config.toml` in git?**
Because it holds your `hf_token` — a Hugging Face access token that grants access to your account. Committing it to a public repository would leak the token. The tracked `config.toml.example` file ships the same structure with safe placeholder values, and `transcript.py` copies it into place on first run.

**What if I already committed `config.toml`?**
Adding it to `.gitignore` does not remove it from git history. Run `git rm --cached config/config.toml` and commit; then **rotate your Hugging Face token immediately** at <https://huggingface.co/settings/tokens>, because the old one is still reachable from history.

**Do the speaker labels match across files?**
No. `SPEAKER_00` in file A is not the same voice as `SPEAKER_00` in file B. Diarization is per-file.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for the full text.

Copyright (c) 2026 Metumxs

---

## 🙏 Acknowledgements

Developed with assistance from **Gemini** and **DeepSeek** (architecture, CUDA runtime handling, documentation). Reviewed, tested and integrated by the maintainer.

This project is not affiliated with OpenAI, SYSTRAN, or NVIDIA.

---

<p align="center"><i>Built for people who'd rather own their transcripts than rent them.</i></p>
