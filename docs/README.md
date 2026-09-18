# 🎙️ WhisperBlocks
> Local transcription pipeline — from audio to LLM-ready text blocks.

> High-performance, fully local batch audio/video transcription using [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper), optimized for NVIDIA GPUs (CUDA).
> Designed for transcribing Ukrainian lectures, meetings and raw audio into structured text or subtitle files — **no cloud, no API keys, no data leaves your machine during transcription.**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
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
- [Auto-Preprocessing](#-auto-preprocessing)
- [LLM Post-Processing](#-llm-post-processing)
- [Model Selection](#-model-selection)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq)
- [License](#-license)

---

## ✨ Features

- **100% local transcription** — audio never leaves your machine.
- **GPU accelerated** — CUDA 12 + cuDNN via pip wheels, no manual CUDA Toolkit install.
- **Batch processing** — drop a list of files into `config.toml` and walk away.
- **Four output modes** — timestamped text, clean plain text, `.srt` subtitles, or all at once.
- **Filler removal** — hints the model to drop verbal tics (`ну`, `типу`, `от`, long pauses) during transcription.
- **Auto-preprocessing** — raw timestamped output is automatically merged into fixed-duration blocks (60 s default) for downstream AI processing.
- **Ready-made LLM prompts** — bundled system prompts (Ukrainian + English) for cleaning up transcripts without paraphrasing.
- **Model-aware filenames** — every output file carries the model name (`--large-v3-turbo`) so you can compare runs.
- **Zero-config launcher** — double-click `run.bat`.

---

## 🖥️ Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10 (21H2) | Windows 11 |
| **GPU** | NVIDIA, 6 GB VRAM | NVIDIA, 8 GB+ VRAM (RTX 3060 / 4060 or better) |
| **Python** | 3.11+ (needs native `tomllib`) | 3.11 / 3.12 |
| **Disk** | ~3 GB free (models + runtimes) | SSD with ~10 GB free |
| **CPU-only fallback** | any modern x86-64 | — (works, but 10–20× slower) |

> **No NVIDIA GPU?** Set `device = "cpu"` in `config.toml` and use a smaller model such as `small` or `medium`.

---

## 📦 Installation

Open a terminal in the project folder and run:

```bash
# 1. Core transcription engine + terminal formatting
pip install faster-whisper rich

# 2. CUDA 12 & cuDNN runtime wheels for Windows
#    (avoids installing the full NVIDIA CUDA Toolkit manually)
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Optional but recommended — create an isolated environment first:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
```

---

## 🗂️ Project Structure

```text
whisper-cli/
├── run.bat                    ← One-click launcher (double-click this)
├── config/
│   └── config.toml            ← User configuration: files, options, hardware
├── src/
│   ├── transcript.py          ← Main transcription script (auto-DLL injection)
│   └── preprocess.py          ← Merges raw segments into fixed-duration blocks
├── prompts/
│   ├── UK-llm-ai-processing-prompt.txt    ← Ukrainian cleanup prompt
│   └── EN-llm-ai-processing-prompt.txt    ← English cleanup prompt
├── docs/
│   ├── README.md              ← This file
│   └── LICENSE                ← MIT license text
│
├── zvvb_pz1_meet.wav          ← Your audio files live here (project root)
└── ...                        ← Output files land here too (see below)
```

**Key rule:** audio files and generated outputs live in the **project root** (next to `run.bat`). The `config/` and `src/` folders are for tooling only.

---

## ⚙️ Configuration

Open `config/config.toml` and edit before running.

```toml
# ── Files to transcribe ─────────────────────────────────────────────
# Relative or absolute paths. Use forward slashes '/' on Windows.
# Relative paths are resolved against the PROJECT ROOT (where run.bat lives),
# NOT against the config/ folder.
files = [
    "lecture.mp4",
    "meeting.wav",
    "C:/Audio/interview.m4a"
]

# ── Output mode ─────────────────────────────────────────────────────
#   "timecode" → timecode-{name}--{model}.txt (with timestamps)
#   "simple"   → simple-{name}--{model}.txt   (plain text)
#   "srt"      → {name}--{model}.srt          (subtitles only)
#   "all"      → all three files above
output_mode = "all"

# ── Filler removal ──────────────────────────────────────────────────
# Biases the decoder via initial prompt to drop verbal tics.
remove_fillers = true

# ── Auto-preprocessing ──────────────────────────────────────────────
# After transcription, merge each timecode-*.txt into N-second blocks.
# Result: timecode-{name}--{model}.blocks.txt
# Set to 0 to disable.
auto_preprocess_block_seconds = 60

# ── Model & hardware ────────────────────────────────────────────────
model_size = "large-v3-turbo"  # Best quality/speed balance
device     = "cuda"            # "cuda" for GPU, "cpu" for CPU-only
language   = "uk"              # ISO-639-1 code, or "auto" for detection
```

### Option reference

| Key | Type | Accepted values | Default |
|-----|------|-----------------|---------|
| `files` | array | Any relative/absolute media path | `[]` |
| `output_mode` | string | `timecode` \| `simple` \| `srt` \| `all` | `all` |
| `remove_fillers` | bool | `true` \| `false` | `true` |
| `auto_preprocess_block_seconds` | number | `0` (off) or any positive number | `60` |
| `model_size` | string | See [Model Selection](#-model-selection) | `large-v3-turbo` |
| `device` | string | `cuda` \| `cpu` | `cuda` |
| `language` | string | ISO-639-1 code, or `auto` | `uk` |

> **On `language = "auto"`:** Whisper's language detector is unreliable for Ukrainian — it frequently classifies short or noisy segments as Russian. Use `"auto"` only for genuinely mixed-language batches. For Ukrainian-only audio, always set `"uk"` explicitly.

---

## ▶️ Usage

1. Drop your audio/video files into the **project root** (the folder with `run.bat`).
2. List them in `config/config.toml` under `files = [...]`.
3. Double-click **`run.bat`** — or run from a terminal:

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
| `RuntimeError: Library cublas64_12.dll is not found` | NVIDIA runtime wheels missing | `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` — `transcript.py` injects the DLL paths automatically via `os.add_dll_directory()` |
| `RuntimeError: Library cudnn_ops_infer64_8.dll is not found` | Same as above | Same as above |
| `CUDA out of memory` | Model too large for VRAM | Use `large-v3-turbo` or `medium`, or set `device = "cpu"` |
| `ModuleNotFoundError: tomllib` | Python < 3.11 | Upgrade to Python 3.11+ |
| `config.toml not found` | Config not in `config/` | Ensure the file is at `config/config.toml` |
| `preprocess.py not importable` | Missing file in `src/` | Ensure `src/preprocess.py` exists next to `src/transcript.py` |
| `✕ Skipped: file not found` | Audio not in project root | Relative paths in `files` are resolved from the project root |
| Transcription is extremely slow | Running on CPU | Confirm `device = "cuda"` |
| Garbled or mixed-language output | Wrong `language` value | Set `language = "uk"` explicitly instead of `"auto"` |
| First run hangs for several minutes | Model download in progress | Expected — ~1.5 GB is fetched once to `~/.cache/huggingface/hub` |

---

## ❓ FAQ

**Does it work offline?**
Transcription — yes, after the first run downloads the model weights. LLM post-processing — depends on your LLM; a local model keeps everything offline, ChatGPT/Claude don't.

**Where are the models stored?**
`~/.cache/huggingface/hub` (`C:\Users\<You>\.cache\huggingface\hub` on Windows).

**Can I transcribe multiple files at once?**
Yes. List every file in `files = [...]`; they are processed sequentially.

**Does it work on AMD or Intel GPUs?**
Not with CUDA. Set `device = "cpu"` — it works, but expect a large slowdown.

**Which video containers are supported?**
Anything `ffmpeg` can decode — `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg` and more.

**Why is the model name in the output filename?**
So you can run the same audio through different models and compare the results directly.

**Can I skip the auto-preprocessing step?**
Yes. Set `auto_preprocess_block_seconds = 0` in the config, or run `python src/preprocess.py` manually later.

**Can I use a different LLM for cleanup?**
Yes — the prompts are plain text, model-agnostic. They work with any chat LLM. For fully offline processing, use a local model via Ollama or LM Studio with a 30B+ parameter model.

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