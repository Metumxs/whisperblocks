import os
import shutil
import sys
from pathlib import Path

# ── Project layout ────────────────────────────────────────────────────
# src/transcript.py → SRC_DIR = .../src, PROJECT_ROOT = .../
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.toml"
CONFIG_EXAMPLE_PATH = PROJECT_ROOT / "config" / "config.toml.example"

# Make sibling modules (preprocess.py) importable when running from any CWD
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── CUDA DLL injection ────────────────────────────────────────────────
# ctranslate2 (used by WhisperX under the hood) and torch both depend on
# cuBLAS/cuDNN DLLs. The nvidia-*-cu12 wheels ship those DLLs inside
# site-packages; exposing their directories to the Windows loader is
# enough — no system-wide CUDA Toolkit install is required.
for path in sys.path:
    cublas_bin = os.path.join(path, "nvidia", "cublas", "bin")
    cudnn_bin = os.path.join(path, "nvidia", "cudnn", "bin")
    if os.path.exists(cublas_bin):
        os.environ["PATH"] = f"{cublas_bin};{cudnn_bin};" + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(cublas_bin)
            os.add_dll_directory(cudnn_bin)
        break

import gc
import time
import tomllib

import torch
import whisperx

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

def format_timestamp(seconds: float, srt_format: bool = False) -> str:
    """Format seconds as MM:SS / HH:MM:SS or SRT timecode."""
    millis = int((seconds - int(seconds)) * 1000)
    sec = int(seconds)
    hours = sec // 3600
    minutes = (sec % 3600) // 60
    secs = sec % 60
    if srt_format:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def display_dashboard(config: dict, files: list) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Model:", config.get("model_size", "large-v3-turbo"))
    table.add_row("Device:", config.get("device", "cuda").upper())
    table.add_row("Output mode:", config.get("output_mode", "all"))
    table.add_row("Language:", config.get("language", "uk"))
    table.add_row("Remove fillers:", "on" if config.get("remove_fillers") else "off")
    table.add_row("Diarization:", "on" if config.get("diarize") else "off")
    auto_pp = config.get("auto_preprocess_block_seconds", 60)
    table.add_row(
        "Auto-preprocess:",
        "off" if not auto_pp else f"on (~{int(auto_pp)}s blocks)",
    )
    table.add_row("Files:", str(len(files)))

    console.print(Panel(table, title="[bold green]WhisperBlocks[/bold green]", border_style="green"))

def resolve_outputs(mode: str) -> tuple[bool, bool, bool]:
    """Return (write_timecoded, write_simple, write_srt) for a given mode."""
    mode = mode.lower()
    return (
        mode in ("timecode", "all"),
        mode in ("simple", "all"),
        mode in ("srt", "all"),
    )

def resolve_media_path(p: str) -> Path:
    """Resolve a media path from config: absolute stays, relative is from PROJECT_ROOT."""
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path

def output_path_for(target: Path, kind: str, model_size: str) -> Path:
    """Build an output path next to the source file.

        timecode → timecode-{stem}--{model}.txt
        simple   → simple-{stem}--{model}.txt
        srt      → {stem}--{model}.srt
    """
    stem = target.stem
    if kind == "timecode":
        return target.with_name(f"timecode-{stem}--{model_size}.txt")
    if kind == "simple":
        return target.with_name(f"simple-{stem}--{model_size}.txt")
    if kind == "srt":
        return target.with_name(f"{stem}--{model_size}.srt")
    raise ValueError(f"Unknown output kind: {kind}")

def run_preprocess(timecoded_files: list[Path], block_seconds: float) -> None:
    """Import preprocess.py and merge each timecoded file into blocks."""
    console.rule("[bold cyan]Preprocessing timecoded outputs[/bold cyan]")

    try:
        import preprocess  # sibling module in src/
    except ImportError as e:
        console.print(f"[bold yellow]Warning:[/bold yellow] preprocess.py not importable ({e}). Skipping.")
        return

    for tc in timecoded_files:
        try:
            out = preprocess.process_file(tc, block_seconds=block_seconds, outdir=None)
            size_kb = out.stat().st_size / 1024
            console.print(f"[green]✓[/green] {tc.name} → [bold]{out.name}[/bold] ({size_kb:.1f} KB)")
        except Exception as e:
            console.print(f"[bold red]✕[/bold red] {tc.name}: {e}")

def ensure_config() -> bool:
    """Ensure config/config.toml exists, seeding it from the example if not.

    Returns True when the caller may proceed to read the config, and
    False when the user needs to edit a freshly-created file first.

    config.toml is gitignored because it holds a Hugging Face access
    token (hf_token) whenever diarization is enabled. The committed
    template lives at config/config.toml.example.
    """
    if CONFIG_PATH.exists():
        return True

    if not CONFIG_EXAMPLE_PATH.exists():
        console.print(
            f"[bold red]Error:[/bold red] [yellow]{CONFIG_PATH}[/yellow] not found, "
            f"and no template to copy from at [yellow]{CONFIG_EXAMPLE_PATH}[/yellow].\n"
            "Re-extract the project archive — the example file is missing."
        )
        return False

    shutil.copyfile(CONFIG_EXAMPLE_PATH, CONFIG_PATH)

    console.print(Panel(
        f"[bold green]Created[/bold green] [yellow]{CONFIG_PATH}[/yellow] "
        f"from the bundled template.\n\n"
        f"Open it and fill in at least:\n"
        f"  [cyan]files[/cyan]      — one or more media paths to transcribe\n"
        f"  [cyan]language[/cyan]   — ISO-639-1 code: \"uk\", \"en\", \"ru\", ...\n"
        f"  [cyan]hf_token[/cyan]   — only if you set diarize = true\n\n"
        f"Then re-run [bold]run.bat[/bold].",
        title="[bold yellow]First run — configuration created[/bold yellow]",
        border_style="yellow",
    ))
    return False

def flush_vram() -> None:
    """Force a Python GC pass and drop cached CUDA allocations.

    WhisperX runs three independent models (ASR, alignment, diarization).
    On an 8 GB card they cannot all coexist in VRAM, so the caller must
    ``del`` the previous model reference before calling this — otherwise
    the tensor memory is still alive and ``empty_cache`` has nothing to
    release.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def transcribe_file(target: Path, config: dict) -> dict:
    """Run the three-stage WhisperX pipeline on a single media file.

    Stage 1 — ASR transcription (WhisperX batched decoding)
    Stage 2 — Forced alignment against a language-specific phoneme model
    Stage 3 — Optional speaker diarization (pyannote.audio)

    Returns a WhisperX result dict with ``segments``, and — when
    diarization is enabled — a ``speaker`` key on each segment.
    """
    model_size = config.get("model_size", "large-v3-turbo")
    device = config.get("device", "cuda")
    language_cfg = str(config.get("language", "uk"))
    language = None if language_cfg.lower() == "auto" else language_cfg
    do_diarize = bool(config.get("diarize", False))
    hf_token = config.get("hf_token", "") or None
    min_speakers = int(config.get("min_speakers", 0) or 0)
    max_speakers = int(config.get("max_speakers", 0) or 0)

    # float16 is the CUDA sweet spot; CPU paths need int8 (faster-whisper
    # rejects float16 on CPU in most builds).
    compute_type = "float16" if device == "cuda" else "int8"

    # Initial prompt primes the decoder for clean language-specific text
    # and, when remove_fillers is on, hints the model to drop verbal tics.
    # NOTE: only the Ukrainian prompt is bundled — extend this block if
    # the tool is used for other languages.
    initial_prompt = (
        "Це лекція. Грамотний текст українською мовою без слів-паразитів: "
        if config.get("remove_fillers") else None
    )
    asr_options: dict = {"beam_size": 5}
    if initial_prompt:
        asr_options["initial_prompt"] = initial_prompt

    # ── Stage 1 — ASR ───────────────────────────────────────────────────
    with console.status(f"[bold yellow]1/3 Loading WhisperX model '{model_size}'...", spinner="dots"):
        model = whisperx.load_model(
            model_size,
            device,
            compute_type=compute_type,
            language=language,
            asr_options=asr_options,
        )

    audio = whisperx.load_audio(str(target))

    with console.status("[bold blue]1/3 Transcribing...", spinner="arc"):
        result = model.transcribe(audio, batch_size=8, language=language)

    detected_lang = result.get("language") or language or "uk"

    # Release the ASR model before loading the aligner — 8 GB cards need
    # the headroom.
    del model
    flush_vram()

    # ── Stage 2 — Alignment ─────────────────────────────────────────────
    with console.status(f"[bold cyan]2/3 Aligning words ({detected_lang})...", spinner="arc"):
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=detected_lang, device=device
            )
            aligned = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            result["segments"] = aligned["segments"]
            del align_model, align_metadata
            flush_vram()
        except Exception as e:
            # Alignment is optional — if the language has no phoneme model
            # available, the raw segment timestamps are still usable.
            console.print(
                f"[bold yellow]Warning:[/bold yellow] alignment skipped ({e})."
            )

    # ── Stage 3 — Diarization ───────────────────────────────────────────
    if do_diarize:
        if not hf_token:
            console.print(
                "[bold yellow]Warning:[/bold yellow] diarize = true but hf_token "
                "is empty in config.toml — skipping speaker diarization."
            )
        else:
            with console.status("[bold magenta]3/3 Diarizing speakers (pyannote)...", spinner="bouncingBar"):
                try:
                    # WhisperX 3.3.4+ removed DiarizationPipeline from the
                    # top-level whisperx namespace (PR #1128). It now lives
                    # in whisperx.diarize. The constructor parameter is
                    # `token` (not the older `use_auth_token`), and since
                    # WhisperX 3.8.0 the default model is pyannote-audio v4's
                    # `pyannote/speaker-diarization-community-1`.
                    from whisperx.diarize import DiarizationPipeline

                    diarize_model = DiarizationPipeline(
                        token=hf_token,
                        device=device,
                    )
                    diarize_segments = diarize_model(
                        audio,
                        min_speakers=min_speakers if min_speakers > 0 else None,
                        max_speakers=max_speakers if max_speakers > 0 else None,
                    )
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    del diarize_model
                    flush_vram()
                except Exception as e:
                    console.print(
                        f"[bold red]✕ Diarization failed:[/bold red] {e}\n"
                        "  Check that the pyannote license agreements are accepted "
                        "on Hugging Face and that hf_token is valid."
                    )

    # Report duration and detected language after all stages finish.
    last_end = result["segments"][-1].get("end", 0.0) if result.get("segments") else 0.0
    console.print(
        f"[dim]Duration: {format_timestamp(last_end)} | "
        f"Language: {detected_lang}[/dim]\n"
    )

    return result

def write_outputs(
    result: dict,
    target: Path,
    model_size: str,
    write_timecoded: bool,
    write_simple: bool,
    write_srt: bool,
) -> Path | None:
    """Write timecode/simple/srt files for one transcribed media file.

    Returns the timecoded path (or None) so the caller can schedule the
    auto-preprocess step.
    """
    tc_path = output_path_for(target, "timecode", model_size) if write_timecoded else None
    simple_path = output_path_for(target, "simple", model_size) if write_simple else None
    srt_path = output_path_for(target, "srt", model_size) if write_srt else None

    tc_file = open(tc_path, "w", encoding="utf-8") if tc_path else None
    simple_file = open(simple_path, "w", encoding="utf-8") if simple_path else None
    srt_file = open(srt_path, "w", encoding="utf-8") if srt_path else None

    try:
        for s_idx, segment in enumerate(result["segments"], start=1):
            t_start = format_timestamp(segment["start"])
            t_end = format_timestamp(segment["end"])
            text = segment["text"].strip()

            # Prefix each segment with the diarized speaker, if available.
            speaker_tag = f"[{segment['speaker']}] " if segment.get("speaker") else ""
            body = f"{speaker_tag}{text}"

            if write_timecoded:
                console.print(
                    Text(f"[{t_start} -> {t_end}]", style="bold yellow"),
                    Text(speaker_tag, style="bold magenta") if speaker_tag else Text(""),
                    text,
                )
            else:
                console.print(f"[bold green]•[/bold green] {body}")

            # Format must match preprocess.py's SEGMENT_RE: the timestamp
            # block comes first, the rest of the line is body text.
            if tc_file:
                tc_file.write(f"[{t_start} -> {t_end}] {body}\n")
            if simple_file:
                simple_file.write(f"{body}\n")
            if srt_file:
                srt_file.write(f"{s_idx}\n")
                srt_file.write(
                    f"{format_timestamp(segment['start'], srt_format=True)} --> "
                    f"{format_timestamp(segment['end'], srt_format=True)}\n"
                )
                srt_file.write(f"{body}\n\n")
    finally:
        for handle in (tc_file, simple_file, srt_file):
            if handle:
                handle.close()

    return tc_path

def main() -> None:
    if not ensure_config():
        return

    with open(CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)

    files = config.get("files", [])
    if not files:
        console.print(
            "[bold yellow]Warning:[/bold yellow] No files listed in config.toml.\n"
            f"  Open [yellow]{CONFIG_PATH}[/yellow] and add at least one path "
            "under [cyan]files = [...][/cyan]."
        )
        return

    display_dashboard(config, files)

    model_size = config.get("model_size", "large-v3-turbo")
    output_mode = config.get("output_mode", "all").lower()
    auto_preprocess_seconds = float(config.get("auto_preprocess_block_seconds", 60))
    write_timecoded, write_simple, write_srt = resolve_outputs(output_mode)

    generated_timecoded: list[Path] = []

    for idx, file_path in enumerate(files, start=1):
        target = resolve_media_path(file_path)
        if not target.exists():
            console.print(f"[bold red]✕ Skipped:[/bold red] [yellow]{file_path}[/yellow] not found.")
            continue

        console.rule(f"[bold cyan]File [{idx}/{len(files)}]: {target.name}[/bold cyan]")
        start_time = time.time()

        try:
            result = transcribe_file(target, config)
        except Exception as e:
            console.print(f"[bold red]✕ Failed:[/bold red] {target.name}: {e}")
            continue

        tc_path = write_outputs(
            result, target, model_size,
            write_timecoded, write_simple, write_srt,
        )
        if tc_path:
            generated_timecoded.append(tc_path)

        elapsed = time.time() - start_time
        console.print(
            f"\n[bold green]✓ Done in {elapsed:.1f}s[/bold green]\n"
        )

    if auto_preprocess_seconds > 0 and generated_timecoded:
        run_preprocess(generated_timecoded, auto_preprocess_seconds)

    console.print(Panel("[bold green]All tasks completed.[/bold green]", border_style="bright_green"))

if __name__ == "__main__":
    main()
