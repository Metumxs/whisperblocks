import os
import sys
from pathlib import Path

# ── Project layout ────────────────────────────────────────────────────
# src/transcript.py → SRC_DIR = .../src, PROJECT_ROOT = .../
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.toml"

# Make sibling modules (preprocess.py) importable when running from any CWD
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ── CUDA DLL injection ────────────────────────────────────────────────
for path in sys.path:
    cublas_bin = os.path.join(path, "nvidia", "cublas", "bin")
    cudnn_bin = os.path.join(path, "nvidia", "cudnn", "bin")
    if os.path.exists(cublas_bin):
        os.environ["PATH"] = f"{cublas_bin};{cudnn_bin};" + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(cublas_bin)
            os.add_dll_directory(cudnn_bin)
        break

import time
import tomllib

from faster_whisper import WhisperModel
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


def main() -> None:
    if not CONFIG_PATH.exists():
        console.print(f"[bold red]Error:[/bold red] [yellow]{CONFIG_PATH}[/yellow] not found.")
        return

    with open(CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)

    files = config.get("files", [])
    if not files:
        console.print("[bold yellow]Warning:[/bold yellow] No files listed in config.toml.")
        return

    display_dashboard(config, files)

    model_size = config.get("model_size", "large-v3-turbo")
    device = config.get("device", "cuda")
    output_mode = config.get("output_mode", "all").lower()
    remove_fillers = config.get("remove_fillers", False)
    auto_preprocess_seconds = float(config.get("auto_preprocess_block_seconds", 60))

    # language = "auto" → None (faster-whisper auto-detects per file).
    # Otherwise the explicit ISO-639-1 code is forced.
    raw_lang = config.get("language", "uk")
    language = None if str(raw_lang).lower() == "auto" else raw_lang

    write_timecoded, write_simple, write_srt = resolve_outputs(output_mode)

    with console.status(f"[bold yellow]Loading model '{model_size}' into VRAM...", spinner="dots"):
        model = WhisperModel(model_size, device=device, compute_type="float16")

    console.print("[bold green]✓[/bold green] Model ready.\n")

    # Initial prompt is language-specific: written in Ukrainian to prime
    # the decoder for clean Ukrainian output.
    initial_prompt = (
        "Це лекція. Грамотний текст українською мовою без слів-паразитів: "
        if remove_fillers else None
    )

    generated_timecoded: list[Path] = []

    for idx, file_path in enumerate(files, start=1):
        target = resolve_media_path(file_path)
        if not target.exists():
            console.print(f"[bold red]✕ Skipped:[/bold red] [yellow]{file_path}[/yellow] not found.")
            continue

        console.rule(f"[bold cyan]File [{idx}/{len(files)}]: {target.name}[/bold cyan]")
        start_time = time.time()

        with console.status("[bold blue]Analyzing spectrogram and running VAD...", spinner="arc"):
            segments, info = model.transcribe(
                str(target),
                language=language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                initial_prompt=initial_prompt,
            )

        console.print(
            f"[dim]Duration: {format_timestamp(info.duration)} | "
            f"Language: {info.language} ({info.language_probability:.1%})[/dim]\n"
        )

        tc_path = output_path_for(target, "timecode", model_size) if write_timecoded else None
        simple_path = output_path_for(target, "simple", model_size) if write_simple else None
        srt_path = output_path_for(target, "srt", model_size) if write_srt else None

        tc_file = open(tc_path, "w", encoding="utf-8") if tc_path else None
        simple_file = open(simple_path, "w", encoding="utf-8") if simple_path else None
        srt_file = open(srt_path, "w", encoding="utf-8") if srt_path else None

        try:
            for s_idx, segment in enumerate(segments, start=1):
                t_start = format_timestamp(segment.start)
                t_end = format_timestamp(segment.end)
                text = segment.text.strip()

                if write_timecoded:
                    console.print(Text(f"[{t_start} -> {t_end}]", style="bold yellow"), text)
                else:
                    console.print(f"[bold green]•[/bold green] {text}")

                if tc_file:
                    tc_file.write(f"[{t_start} -> {t_end}] {text}\n")
                if simple_file:
                    simple_file.write(f"{text}\n")
                if srt_file:
                    srt_file.write(f"{s_idx}\n")
                    srt_file.write(
                        f"{format_timestamp(segment.start, srt_format=True)} --> "
                        f"{format_timestamp(segment.end, srt_format=True)}\n"
                    )
                    srt_file.write(f"{text}\n\n")
        finally:
            for handle in (tc_file, simple_file, srt_file):
                if handle:
                    handle.close()

        if tc_path:
            generated_timecoded.append(tc_path)

        elapsed = time.time() - start_time
        speed = info.duration / elapsed if elapsed > 0 else 0
        console.print(
            f"\n[bold green]✓ Done in {elapsed:.1f}s[/bold green] "
            f"[dim]({speed:.1f}× realtime)[/dim]\n"
        )

    if auto_preprocess_seconds > 0 and generated_timecoded:
        run_preprocess(generated_timecoded, auto_preprocess_seconds)

    console.print(Panel("[bold green]All tasks completed.[/bold green]", border_style="bright_green"))


if __name__ == "__main__":
    main()