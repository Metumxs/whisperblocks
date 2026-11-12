# 🎙️ WhisperBlocks
> Local transcription pipeline — from audio to LLM-ready text blocks.

> High-performance, fully local batch audio/video transcription using [`WhisperX`](https://github.com/m-bain/whisperX), optimized for NVIDIA GPUs (CUDA).
> Designed for transcribing Ukrainian lectures, meetings and raw audio into structured text or subtitle files — **no transcription API and no media uploads**. Model weights may be downloaded on the first real run; optional gated diarization models need a Hugging Face access token. Once all required models are cached, transcription can run locally offline.

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
- **Word-level alignment** — forced alignment refines word/segment boundaries internally; SRT still contains segment-level cues, not one cue per word.
- **Optional speaker diarization** — pyannote.audio v4 tags every segment with a speaker label; runs as a third pipeline stage.
- **Batch processing** — drop a list of files into `config.toml` and walk away.
- **Four output modes** — timestamped text, clean plain text, `.srt` subtitles, or all at once.
- **Filler removal** — hints the model to drop verbal tics (`ну`, `типу`, `от`, long pauses) during transcription.
- **Auto-preprocessing** — raw timestamped output is automatically merged into fixed-duration blocks (60 s default) for downstream AI processing.
- **Protected results** — skip existing outputs by default; opt in to replacement with `--overwrite`.
- **Offline preflight** — validate paths and inspect the exact plan with `--dry-run`; `--check-config` also reports CUDA status without loading models.
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

> **No NVIDIA GPU?** Set `device = "cpu"` in `config.toml` and use a smaller model such as `small` or `medium`. Note that `install.bat` still installs the CUDA build of PyTorch — the installer intentionally verifies that a CUDA-enabled wheel is installed, even on CPU-only hosts.

> **Why Python 3.12 specifically?** It is the project's supported runtime for the pinned WhisperX / PyTorch stack. `install.bat` checks both the launcher and any reused `.venv`; a different major.minor version is rejected without deleting it. Other Python versions have not been verified for the production ML workflow here.

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
2. Creates `.venv\` (reuses it only after verifying it is Python 3.12).
3. Upgrades `pip` inside the venv.
4. **Installs CUDA-enabled PyTorch from the PyTorch index** (`cu128`) — this is the critical step.
5. Installs WhisperX and the remaining dependencies from `requirements/requirements.txt`.
6. Verifies the installed torch wheel has a CUDA runtime (`torch.version.cuda is not None`). A CPU-only wheel fails with reinstall instructions. A CUDA-enabled wheel with no active NVIDIA GPU/driver emits a warning and succeeds; set `device = "cpu"` to transcribe on that host.

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
pip install -r requirements/requirements.txt
```

> **Do not skip step 4.** PyPI ships the CPU-only PyTorch build on Windows. Running `pip install -r requirements/requirements.txt` alone produces an environment where WhisperX loads fine but crashes at the first GPU call with `Torch not compiled with CUDA enabled`.

A condensed cheat-sheet of these commands (with recovery steps and a list of safe-to-ignore warnings) ships as `docs/SETUP-NOTES.txt`.

### What gets installed

| Package | Why |
|---------|-----|
| `whisperx==3.8.6` | Transcription + alignment + diarization pipeline |
| `rich` | Terminal dashboard, live segment output, panels |
| `torch==2.8.0` + `torchaudio==2.8.0` | Neural network runtime (CUDA build from the PyTorch index) |
| `nvidia-cublas-cu12` | cuBLAS runtime DLLs (Windows; avoids a full CUDA Toolkit install) |
| `nvidia-cudnn-cu12` | cuDNN runtime DLLs (Windows) |

> `transcript.py` registers the existing CUDA DLL paths from these wheels before importing the ML runtime on Windows, retaining their loader handles for the process lifetime — you do **not** need to install the NVIDIA CUDA Toolkit separately.

### Install FFmpeg for media decoding

WhisperX media loading uses FFmpeg. Install it and make sure `ffmpeg -version` works in the terminal used to launch the project. A separate pyannote/torchcodec decoder warning may also depend on matching decoder libraries; do not assume every decoder error is harmless:

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
WhisperBlocks/
├── .gitignore
├── install.bat
├── run.bat
├── requirements/
│   └── requirements.txt
├── config/
│   ├── config.toml.example       ← safe template
│   └── config.toml               ← local only; created on first run
├── input/
│   └── .gitkeep                  ← media contents ignored
├── output/
│   ├── .gitkeep                  ← generated contents ignored
│   └── <source-folder>/          ← transcripts, blocks, source ownership marker
├── src/
│   ├── transcript.py             ← ASR → alignment → optional diarization
│   ├── preprocess.py             ← greedy fixed-duration blocks
│   └── file_io.py                ← per-file atomic UTF-8 writes
├── tests/                        ← standard-library unittest suite
└── docs/
    ├── README.md
    ├── SETUP-NOTES.txt
    └── LICENSE
```

`input/` and `output/` are created automatically on startup. Place new media in
`input/`; nothing moves existing media or legacy transcripts automatically.
The local `.venv/` created by `install.bat` is not tracked.

---

## ⚙️ Configuration

### How `config.toml` works

WhisperBlocks keeps **two** config files in `config/`:

| File | Tracked in git? | Who edits it |
|------|-----------------|--------------|
| `config.toml` | ❌ No (gitignored) | You — this is your live config |
| `config.toml.example` | ✅ Yes | Nobody — it is a template |

`config.toml` is gitignored because it holds your Hugging Face access token (`hf_token`) whenever speaker diarization is enabled. A token pasted into a public repository is a leaked token — revoke/rotate it immediately; do not rely on automatic detection.

**On first run** (when `config/config.toml` does not exist), `transcript.py` copies `config.toml.example` into place, prints a hint panel, and exits. You then edit `config.toml` to add your file list and re-run `run.bat`. This keeps new live configs ignored by default; it cannot untrack an already committed config or prevent a deliberate force-add.

**Never edit `config.toml.example`.** Your changes to it will be ignored — it only exists to seed `config.toml` on first run.

> **Already committed `config.toml` before this pattern existed?** Adding it to `.gitignore` does not remove it from git history. Run `git rm --cached config/config.toml` and commit, then rotate your Hugging Face token immediately.

### Editing `config.toml`

```toml
# ── Files to transcribe ─────────────────────────────────────────────
files = [
    "input/lecture.mp4",
    "input/meeting.wav",
    "C:/Audio/interview.m4a"
]

# ── Output mode ─────────────────────────────────────────────────────
output_mode = "all"

# ── Decoder hint (used only when remove_fillers = true) ──────────────
remove_fillers = true
initial_prompt = "Це лекція. Грамотний текст українською мовою без слів-паразитів: "

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
batch_size = 8
```

### Option reference

| Key | Type | Accepted values | Fallback (template if different) |
|-----|------|-----------------|---------|
| `files` | array of non-empty strings | Relative/absolute media paths; non-empty unless CLI paths are given | `[]` (edit before use) |
| `output_mode` | string | `timecode` \| `simple` \| `srt` \| `all`, case-insensitive | `all` |
| `remove_fillers` | bool | `true` \| `false`; decoder hint only, no regex word deletion | `false` (template: `true`) |
| `initial_prompt` | string | Language-specific decoder hint; used only with `remove_fillers` | empty (template: Ukrainian hint) |
| `auto_preprocess_block_seconds` | finite number | `0` (off) or positive; requires timecode output | `60` |
| `diarize` | bool | `true` \| `false` | `false` |
| `hf_token` | string | HF token; non-empty `HF_TOKEN` environment variable takes precedence | `""` |
| `min_speakers` | integer | `0` (auto) or positive; must not exceed a positive `max_speakers` | `0` |
| `max_speakers` | integer | `0` (auto) or positive | `0` |
| `batch_size` | integer | `>= 1`; existing configs without it retain batch size 8 | `8` |
| `model_size` | non-empty string | No `..`, path separators or Windows-illegal filename characters; no fixed model allowlist | `large-v3-turbo` |
| `device` | string | `cuda` \| `cpu` | `cuda` |
| `language` | string | ISO-639-1 code, or `auto` | `uk` |

> **On `language = "auto"`:** Whisper's language detector is unreliable for Ukrainian — it frequently classifies short or noisy segments as Russian. Use `"auto"` only for genuinely mixed-language batches. For Ukrainian-only audio, always set `"uk"` explicitly.

---

## ▶️ Usage

1. Put new media in **`input/`** (nested folders are welcome).
2. Edit `config/config.toml`, for example `files = ["input/course/lecture.mp3"]`.
3. Double-click **`run.bat`**, or use the CLI with `.venv` active:

   ```bat
   run.bat --check-config
   run.bat --dry-run
   run.bat input\clip.mp4
   run.bat "D:\Audio\my lecture.wav" --overwrite
   python src/transcript.py --config config/other.toml --dry-run input/clip.mp4
   ```

Drag one or more media files onto `run.bat` to pass their paths as positional
arguments. CLI paths **replace**, not append to, the config's `files` list.
`run.bat` forwards arguments and preserves Python's exit code after `pause`.

### Input resolution

- Absolute paths are resolved as given.
- Relative paths first check the project root (not the current working directory).
- If that path is absent and is not already prefixed `input/`, check `input/<path>`.
- If both root and fallback paths exist, the root path wins with an ambiguity
  warning; use an explicit `input/...` path to choose the other file.
- Inputs must be readable regular files. Resolved aliases/symlinks are deduplicated
  with a warning; no media is moved. A directory is an error, not a recursive input.

### Preflight, protection and flags

| Option | Behaviour |
|--------|-----------|
| positional `files...` | Override the config file list; quote paths containing spaces |
| `--config PATH` | Use a custom config; relative paths are from the project root. An explicitly missing file is an error, never seeded |
| `--dry-run` | Dashboard and exact input → output plan, including ready, skipped and invalid sources; no ML imports/models |
| `--check-config` | Same validation/plan plus token-source and CUDA status; may import torch for status, never loads models |
| `--overwrite` | Allow replacement of expected transcript/block files for the targeted sources; never transfer another source's output-directory ownership |

The first run with a missing **default** config still seeds the template and
exits `0`, even with diagnostic flags; edit it and rerun. Diagnostics may create
the empty `input/` and `output/` directories but do not write transcript files.

Configuration types/values are validated before ML imports; booleans are not
accepted as numeric settings. An empty file list is invalid unless CLI files
are supplied. Missing or unreadable inputs are reported together in preflight;
other valid inputs still run. A non-empty `HF_TOKEN` environment variable
(ignoring surrounding whitespace) overrides `hf_token`; only its source/status
is displayed, never its value. Missing tokens with `diarize = true` warn upfront
and disable only diarization. A Cyrillic decoder hint with `language != "uk"`
(including `"auto"`) warns; the tool never rewrites that hint.

If **any** expected active-mode file already exists (including `.blocks.txt`
when enabled), that whole source is skipped **before** transcription unless
`--overwrite` is passed. Even a partial previous run triggers this protection.
Delete unwanted partial files manually or explicitly rerun with `--overwrite`.
No interactive replacement prompts are used. A batch with no runnable sources
never checks CUDA or loads ML dependencies. Diagnostic modes do not fail solely
for a missing CUDA GPU; a real CUDA queue requires a CUDA-capable runtime/GPU
or `device = "cpu"`.

Writes use temporary sibling files, close them, then `os.replace()` each complete
file. Errors/interrupts clean up temporary files. **This is per-file atomicity,
not a multi-file transaction or cross-process lock**; do not run simultaneous
writers for the same source. A late replace failure may leave other requested
files already committed and is reported as `failed_write`.

Exit codes: `0` = success / only existing-output skips / successful first-run
seeding; `1` = any missing/invalid input, runtime, transcription, write or
preprocessing failure; `2` = invalid configuration or CLI syntax. Diagnostics
return `1` for missing/invalid inputs and `0` otherwise. Alignment/diarization
failures that retain a valid transcript are warnings, not batch failures.
The final summary distinguishes `succeeded`, `skipped_existing`,
`missing_or_invalid`, `failed_transcription`, `failed_write`, `failed_preprocess`;
`succeeded` counts sources whose requested preprocessing also succeeded.

---

## 📄 Output Files

All transcript outputs live under **`output/<source-folder>/`**, never next to the media. Filenames keep their existing mode/model conventions. Example inside `output/zvvb_pz1_meet/` for `input/zvvb_pz1_meet.wav` with `large-v3-turbo`:

| File | Description |
|------|-------------|
| `timecode-zvvb_pz1_meet--large-v3-turbo.txt` | Full transcript with `[MM:SS -> MM:SS]` timestamps on every segment |
| `timecode-zvvb_pz1_meet--large-v3-turbo.blocks.txt` | The above, merged into ~60-second blocks (auto-generated) |
| `simple-zvvb_pz1_meet--large-v3-turbo.txt` | Plain text, one segment per line, no timestamps |
| `zvvb_pz1_meet--large-v3-turbo.srt` | Standard `.srt` subtitle file |

When diarization assigns a speaker, the segment body is prefixed with `[SPEAKER_NN]`; the timecode still comes first. SRT cue numbers/timing lines are not prefixed. Block merging preserves these tags inside the merged text.

**Why the model name in the filename?** So you can transcribe the same audio with `medium` and `large-v3-turbo` and compare results side by side without renaming anything.

### Source-folder mapping and collisions

- Sources physically inside `input/` mirror their relative parent folders and
  stem: `input/course/lecture.mp3` → `output/course/lecture/`. This also applies
  when the same input file is passed by an absolute path.
- Root-level and external sources use only a sanitized basename stem:
  `D:/Audio/lecture.mp3` → `output/lecture/`, with no drive/parent directories.
  Symlinks are resolved before deciding whether a source is inside `input/`.
- Unsafe Windows characters, trailing spaces/dots and reserved device names in
  source components are sanitized; ordinary filenames stay unchanged.
- For a collision, sources are considered in stable canonical-path order
  (`input/` sources first), independent of CLI list order. The first available
  natural folder is used, then `<stem>_<extension>`, then a 10-character SHA-256
  suffix of the canonical source path (and a deterministic numeric suffix if
  necessary). Natural folder names requested elsewhere in the batch are reserved.
- A local `.whisperblocks-source` marker records directory ownership before real
  processing. It is ignored by Git along with all output contents, contains only
  the resolved media path (no token), and is not a transcript. Existing ownership
  wins on later runs, so separately processed colliding files do not share folders.
  Keep markers with their output folders; relocating sources may produce a new
  suffixed folder. Old folders without a marker retain the natural mapping and
  skip-existing protection and emit an ownership warning; their historical ownership cannot be inferred.
- Every computed directory/file is checked with `resolve()` to remain strictly
  inside the resolved `output/` root. Escaping symlinks/traversal are errors.
  Legacy transcripts beside media files are never deleted or migrated.

### Format examples

Plain timecode timestamps use **whole seconds** (`MM:SS`, or `HH:MM:SS` for
long recordings). SRT rounds to **milliseconds**, including carry into the next
second. Word-level alignment refines internal boundaries; SRT cues remain
**segment-level**. Duration/language is console-only, not an added file header.

**`timecode`** — segment-by-segment:

```text
[00:00 -> 00:05] Добрий день, сьогодні ми розглянемо...
[00:05 -> 00:11] ...основи машинного навчання.
```

**`simple`** — plain text for pasting into an LLM:

```text
Добрий день, сьогодні ми розглянемо...
...основи машинного навчання.
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

**With successful diarization enabled** (`diarize = true`), segment bodies gain a speaker tag:

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
4. Set a non-empty `HF_TOKEN` environment variable or configure `hf_token` locally in `config/config.toml`, then set `diarize = true`. The environment takes precedence; do not paste real tokens into examples or logs.

> **Using an older WhisperX (< 3.8)?** That version used pyannote-audio 3.x, whose models are `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`. Accept both of those agreements instead. The current `requirements/requirements.txt` pins WhisperX 3.8.6, so you only need the `community-1` agreement.

### Speaker-count hints

If the number of speakers in the audio is known ahead of time, telling the model speeds up diarization and improves accuracy:

- **Unknown count** — leave `min_speakers = 0` and `max_speakers = 0`. The model decides.
- **Exact count** — set both to the same value (e.g. 4 comedians on stage → `min_speakers = 4`, `max_speakers = 4`).
- **Range** — set `min_speakers = 2` and `max_speakers = 5` to bound the search space.

> **Known limitation.** The open-source `community-1` model struggles with fast, overlapping dialogue between speakers with similar voices (interviews, panel shows, comedy). Short test clips make this worse: the model needs several minutes of audio to build stable speaker clusters. For a clip that is only 1–2 minutes long, forcing the exact speaker count via `min_speakers` / `max_speakers` is the single most effective tweak.

### VRAM behaviour

Diarization loads a **third** model (~100 MB, but each stage's peak VRAM is what matters). On an 8 GB card, WhisperX cannot keep the ASR, alignment and diarization models resident simultaneously. `transcript.py` explicitly releases each model (clear model references, then `gc.collect(); torch.cuda.empty_cache()`) in `finally` before loading the next, including stage exceptions — do not remove that cleanup if you edit the pipeline.

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

**Triggered automatically** at the end of a batch for successfully written timecode files when `auto_preprocess_block_seconds > 0`. The code fallback and bundled template both use **60**; an existing live value such as `120` is honored. Modes `simple` / `srt` do not generate timecode files and warn upfront if auto-preprocessing is configured.

### Output format

```text
Source: timecode-zvvb_pz1_meet--large-v3-turbo.txt
Segments: 2 | Blocks: 1 (~60s each)

=== BLOCK 01 | 00:00 → 00:11 ===
Добрий день, сьогодні ми розглянемо... ...основи машинного навчання.
```

### Manual use

The script also works standalone:

```bash
python src/preprocess.py "output/*/timecode-*.txt"
python src/preprocess.py "output/**/timecode-*.txt" --block 90
python src/preprocess.py "output/*/timecode-*.txt" --outdir output/manual-blocks --overwrite
```

The standalone script expands wildcard patterns itself (also Windows
`output\*\timecode-*.txt`, or recursive `output\**\timecode-*.txt`). It excludes
`*.blocks.txt` from wildcard matches, preserves existing literal filenames with brackets, deduplicates resolved paths in order, and
returns non-zero for each unmatched input/pattern. Existing block outputs are
skipped unless `--overwrite` is passed. `--block` must be finite and `> 0`.
Inputs must be readable UTF-8 regular files with at least one timestamped
segment. Starts must be non-decreasing; overlapping segments are accepted.
Malformed bracketed arrow timecodes, negative/out-of-range times and `end < start`
are errors with line numbers; ordinary `[SPEAKER_00]` / `[inaudible]` wrapped
text is preserved. An optional header from another producer is passed through;
WhisperBlocks itself does not generate a duration/language file header.

---

## 🤖 LLM Post-Processing

There is no LLM API integration. Copy a small batch (for example 5–10 blocks)
from `.blocks.txt` into an LLM of your choice and ask it to correct obvious ASR
errors and punctuation without paraphrasing, shortening, inventing content or
changing block boundaries. Verify the result against the source audio before
continuing. Use a local LLM if transcript text must remain offline; pasting text
into a hosted service shares that text with the chosen service.

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
| `Configuration error` mentioning a missing `config.toml.example` | Archive extracted incompletely | Re-extract the project — `config/config.toml.example` must exist |
| `Transcription ... Torch not compiled with CUDA enabled` | PyTorch CPU build was installed instead of the CUDA build | Reinstall: `pip uninstall -y torch torchaudio` then `pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128` |
| `pip check` reports conflicts involving `torch` version | Wrong CUDA index used (cu124 caps at torch 2.6.0) | Use `cu128` (or `cu126`) — WhisperX 3.8.6 needs `torch~=2.8.0` |
| `module 'whisperx' has no attribute 'DiarizationPipeline'` | WhisperX 3.3.4+ moved the class out of the top-level namespace | Already fixed in `src/transcript.py` — it imports from `whisperx.diarize`. If you see this, your `transcript.py` is out of date. |
| `ModuleNotFoundError: No module named 'whisperx'` | Install ran outside the venv | `.venv\Scripts\activate` first, then `pip install -r requirements/requirements.txt` |
| `RuntimeError: Library cublas64_12.dll is not found` | NVIDIA runtime wheels missing | `.venv\Scripts\activate` then `pip install -r requirements/requirements.txt` |
| `RuntimeError: Library cudnn_ops_infer64_8.dll is not found` | Same as above | Same as above |
| `UserWarning: torchcodec is not installed correctly` | Decoder/runtime mismatch or FFmpeg unavailable | Verify `ffmpeg -version` and the installed decoder/runtime; if a stage actually fails, inspect that error rather than assuming fallback |
| `ReproducibilityWarning: TensorFloat-32 (TF32) has been disabled` | pyannote disables TF32 for reproducibility | Benign — informational only |
| `Lightning automatically upgraded your loaded checkpoint` | One-time checkpoint migration | Benign — informational only |
| `huggingface_hub cache uses symlinks ... machine does not support them` | Windows symlink policy | Benign — HF cache works without symlinks. Enable Developer Mode to silence |
| `Xet Storage is enabled ... hf_xet package is not installed` | Optional HF accelerator missing | Benign. Install with `pip install hf_xet` in the venv to speed up downloads |
| `std(): degrees of freedom is <= 0` (pyannote pooling) | Very short segments in the input | Benign — affects short clips more. Use longer audio or set `min_speakers`/`max_speakers` |
| Diarization assigns the wrong speaker to overlapping lines | Known limitation of `community-1` on fast, overlapping dialogue | Try `min_speakers`/`max_speakers` hints, use longer audio, or accept that some lines will be misattributed |
| `CUDA out of memory` at the alignment or diarization stage | Previous model wasn't released | Confirm the `finally` model-release / `flush_vram()` blocks are still present in `src/transcript.py` |
| `CUDA out of memory` at the ASR stage | Model too large for VRAM | Use `large-v3-turbo` or `medium`, or set `device = "cpu"` |
| `401 Client Error` when diarizing | pyannote license not accepted, or token invalid | Accept the license at `pyannote/speaker-diarization-community-1`, regenerate the HF token, update `config.toml` |
| `Diarization failed` warning, script continues | Any diarization error | The pipeline falls through gracefully — transcript is written without speaker tags |
| `alignment skipped` warning, script continues | Language has no phoneme model on HF | Expected for rare languages; segments keep their ASR timestamps |
| `preprocess.py not importable` | Missing file in `src/` | Ensure `src/preprocess.py` exists next to `src/transcript.py` |
| `invalid_or_missing: ... Input not found` | Input path absent or unreadable | Check root-first / `input/` fallback resolution with `--dry-run` |
| `skipped_existing` | At least one expected output exists | Inspect the plan; pass `--overwrite` only if replacement is intended |
| `CUDA is not available` | No CUDA GPU/driver or wrong torch wheel | Check `torch.version.cuda` and driver; use `device = "cpu"` for fallback |
| Existing `.venv` is not Python 3.12 | Wrong environment runtime | Delete `.venv` manually if appropriate and rerun `install.bat`; it is never auto-deleted |
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
Yes — copy `.blocks.txt` batches into whichever LLM you trust. For fully offline cleanup, use a local model and verify its edits manually.

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

## 🧪 Offline Development Checks

From the project root:

```bash
python -m unittest discover -s tests -v
python -m py_compile src/transcript.py src/preprocess.py src/file_io.py tests/support.py tests/test_config.py tests/test_paths.py tests/test_io.py tests/test_preprocess.py tests/test_batch.py tests/test_pipeline.py tests/test_launchers.py
python -c "import pathlib, tomllib; tomllib.loads(pathlib.Path('config/config.toml.example').read_text(encoding='utf-8')); print('TOML OK')"
```

Tests use `unittest`, temporary synthetic files and mocked torch/WhisperX/UI
boundaries; no GPU, real media, model downloads, network or extra test packages.
They never invoke the real installer. Git ignore checks use a temporary Git
repository when Git is available. Python 3.12 is still the production target;
Linux/offline checks are not proof of live Windows/CUDA compatibility.

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
