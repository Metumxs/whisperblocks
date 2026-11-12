import argparse
import gc
import hashlib
import importlib
import math
import os
import re
import sys
import time
import tomllib
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

# ── Project layout ────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.toml"
CONFIG_EXAMPLE_PATH = PROJECT_ROOT / "config" / "config.toml.example"

# Make sibling modules importable when running from any CWD.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from file_io import atomic_text_writer, path_exists
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# ML dependencies are deliberately lazy: invalid configs, empty batches and
# --dry-run never need WhisperX. --check-config may inspect torch, not models.
torch = None
whisperx = None
_DLL_DIRECTORY_HANDLES: list = []
_DLL_DIRECTORIES: set[str] = set()
SOURCE_MARKER = ".whisperblocks-source"

# Fallbacks preserve old configs that omit keys. The bundled template enables
# remove_fillers and supplies its language-specific initial_prompt explicitly.
DEFAULT_CONFIG = {
    "files": [],
    "model_size": "large-v3-turbo",
    "device": "cuda",
    "language": "uk",
    "output_mode": "all",
    "remove_fillers": False,
    "initial_prompt": "",
    "auto_preprocess_block_seconds": 60,
    "diarize": False,
    "hf_token": "",
    "min_speakers": 0,
    "max_speakers": 0,
    "batch_size": 8,
}


def warning(message: str) -> None:
    console.print(Text(f"Warning: {message}", style="yellow"))


def error_detail(error: Exception, config: dict | None = None) -> str:
    """Do not echo access tokens, including tokens embedded in library errors."""
    message = str(error)
    tokens = [os.environ.get("HF_TOKEN", "")]
    if config is not None:
        tokens.append(config.get("hf_token", ""))
    for token in tokens:
        if isinstance(token, str) and token.strip():
            message = message.replace(token, "[redacted]")
    return re.sub(r"hf_[A-Za-z0-9]{8,}", "[redacted]", message)


def setup_cuda_dlls() -> None:
    """Expose existing CUDA wheel directories and retain Windows DLL handles."""
    if not hasattr(os, "add_dll_directory"):
        return
    for path in sys.path:
        for library in ("cublas", "cudnn"):
            directory = os.path.join(path, "nvidia", library, "bin")
            if not os.path.isdir(directory):
                continue
            directory = os.path.abspath(directory)
            key = os.path.normcase(directory)
            if key in _DLL_DIRECTORIES:
                continue
            try:
                handle = os.add_dll_directory(directory)
            except OSError as error:
                warning(f"Cannot register CUDA DLL directory {directory}: {error_detail(error)}")
                continue
            _DLL_DIRECTORY_HANDLES.append(handle)
            _DLL_DIRECTORIES.add(key)
            os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")


def load_torch():
    global torch
    if torch is None:
        setup_cuda_dlls()
        torch = importlib.import_module("torch")
    return torch


def load_whisperx():
    global whisperx
    if whisperx is None:
        setup_cuda_dlls()
        whisperx = importlib.import_module("whisperx")
    return whisperx


def validate_model_size(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model_size: expected a non-empty string")
    if ".." in value or re.search(r'[<>:"/\\|?*\x00-\x1f]', value):
        raise ValueError("model_size: must be filename-safe (no '..', separators or Windows-illegal characters)")
    return value


def validate_config(
    raw: dict, cli_files: list[str] | None = None,
    *, environ: Mapping[str, str] | None = None,
) -> dict:
    """Return validated defaults + config, without printing values or using ML."""
    config = {**DEFAULT_CONFIG, **raw}
    files = config["files"]
    if not isinstance(files, list) or any(
        not isinstance(path, str) or not path.strip() for path in files
    ):
        raise ValueError("files: expected a list of non-empty path strings")
    if cli_files:
        if any(not isinstance(path, str) or not path.strip() for path in cli_files):
            raise ValueError("files: CLI paths must be non-empty strings")
        files = list(cli_files)
    if not files:
        raise ValueError("files: list is empty; add media paths or pass positional CLI files")
    config["files"] = list(files)

    for key in ("output_mode", "device", "language"):
        value = config[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key}: expected a non-empty string")
    config["output_mode"] = config["output_mode"].lower()
    if config["output_mode"] not in {"timecode", "simple", "srt", "all"}:
        raise ValueError("output_mode: expected timecode, simple, srt or all (case-insensitive)")
    if config["device"] not in {"cuda", "cpu"}:
        raise ValueError("device: expected 'cuda' or 'cpu'")
    config["model_size"] = validate_model_size(config["model_size"])

    for key in ("remove_fillers", "diarize"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key}: expected a boolean")
    for key in ("initial_prompt", "hf_token"):
        if not isinstance(config[key], str):
            raise ValueError(f"{key}: expected a string")
    for key in ("min_speakers", "max_speakers", "batch_size"):
        minimum = 1 if key == "batch_size" else 0
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key}: expected an integer >= {minimum} (not a boolean)")
    if (config["min_speakers"] > 0 and config["max_speakers"] > 0
            and config["min_speakers"] > config["max_speakers"]):
        raise ValueError("min_speakers: must not exceed max_speakers when both are > 0")

    seconds = config["auto_preprocess_block_seconds"]
    valid = type(seconds) in (int, float)
    if valid:
        try:
            valid = math.isfinite(seconds) and seconds >= 0
        except OverflowError:
            valid = False
    if not valid:
        raise ValueError("auto_preprocess_block_seconds: expected a finite number >= 0 (not a boolean)")

    environment = os.environ if environ is None else environ
    env_token = environment.get("HF_TOKEN", "").strip()
    config_token = config["hf_token"].strip()
    config["hf_token"] = env_token or config_token
    config["_token_source"] = "HF_TOKEN environment" if env_token else (
        "config file" if config_token else "not configured"
    )
    return config


def configuration_warnings(config: dict) -> list[str]:
    messages: list[str] = []
    if config["diarize"] and not config["hf_token"]:
        messages.append(
            "diarize = true but neither HF_TOKEN nor hf_token is set; "
            "speaker diarization will be skipped (transcription can continue)."
        )
    if (config["auto_preprocess_block_seconds"] > 0
            and config["output_mode"] in {"simple", "srt"}):
        messages.append(
            "auto_preprocess_block_seconds is enabled, but auto-preprocessing only "
            "runs when timecode output is generated (output_mode = timecode or all)."
        )
    if (config["remove_fillers"] and config["initial_prompt"]
            and re.search(r"[\u0400-\u04ff]", config["initial_prompt"])
            and config["language"].lower() != "uk"):
        messages.append(
            "initial_prompt appears Ukrainian/Cyrillic while language is not 'uk' "
            "(including 'auto'); a Ukrainian decoder hint can degrade non-Ukrainian "
            "transcription. Edit the prompt or disable remove_fillers."
        )
    return messages


def ensure_config(config_path: Path | None = None, *, seed: bool = True) -> bool:
    """Seed only a missing default config, exclusively; never replace a live one."""
    config_path = CONFIG_PATH if config_path is None else config_path
    if path_exists(config_path):
        return True
    if not seed:
        raise FileNotFoundError(f"Explicit --config file not found: {config_path}")
    template = CONFIG_EXAMPLE_PATH.read_bytes()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with config_path.open("xb") as handle:
            handle.write(template)
    except FileExistsError:
        return True
    console.print(Panel(
        Text(
            f"Created {config_path} from the bundled template.\n\n"
            "Open it and fill in files (prefer input/...), language, and optionally "
            "hf_token for diarization, or set HF_TOKEN in the environment.\n"
            "Then re-run run.bat. Existing config files are never replaced."
        ),
        title="First run — configuration created", border_style="yellow",
    ))
    return False


def format_timestamp(seconds: float, srt_format: bool = False) -> str:
    """Format whole-second plain timestamps or correctly rounded SRT timecodes."""
    if srt_format:
        total_ms = int(round(max(0.0, float(seconds)) * 1000))
        sec, millis = divmod(total_ms, 1000)
    else:
        sec = int(seconds)
    hours, remainder = divmod(sec, 3600)
    minutes, secs = divmod(remainder, 60)
    if srt_format:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def display_dashboard(config: dict, files: list[str]) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="bold white")
    for key, value in (
        ("Model:", config["model_size"]),
        ("Device:", config["device"].upper()),
        ("Output mode:", config["output_mode"]),
        ("Language:", config["language"]),
        ("Batch size:", str(config["batch_size"])),
        ("Remove fillers:", "on" if config["remove_fillers"] else "off"),
        ("Diarization:", "on" if config["diarize"] else "off"),
        ("HF token:", config["_token_source"]),
        ("Auto-preprocess:", "off" if not config["auto_preprocess_block_seconds"] else
            f"on (~{config['auto_preprocess_block_seconds']:g}s blocks)"),
        ("Files:", str(len(files))),
    ):
        table.add_row(key, Text(value))
    console.print(Panel(table, title="[bold green]WhisperBlocks[/bold green]", border_style="green"))


def resolve_outputs(mode: str) -> tuple[bool, bool, bool]:
    """Return (write_timecoded, write_simple, write_srt); reject unknown modes."""
    mode = mode.lower()
    if mode not in {"timecode", "simple", "srt", "all"}:
        raise ValueError("output_mode: expected timecode, simple, srt or all")
    return mode in {"timecode", "all"}, mode in {"simple", "all"}, mode in {"srt", "all"}


def project_root_path(project_root: Path | None = None) -> Path:
    return (PROJECT_ROOT if project_root is None else project_root).resolve()


def resolve_media_path(
    p: str, *, project_root: Path | None = None,
    warn: Callable[[str], None] = warning,
) -> Path:
    """Root-first relative lookup, then input/ fallback; return a physical path."""
    root = project_root_path(project_root)
    path = Path(p.replace("\\", os.sep) if os.name != "nt" else p)
    if path.is_absolute():
        return path.resolve()
    primary = root / path
    explicit_input = bool(path.parts) and path.parts[0].lower() == "input"
    if not explicit_input:
        fallback = root / "input" / path
        if primary.exists():
            if fallback.exists():
                warn(
                    f"Ambiguous relative input '{p}': both {primary} and {fallback} exist. "
                    "Using the project-root path; specify 'input/...' explicitly for the other file."
                )
        elif fallback.exists():
            return fallback.resolve()
    return primary.resolve()


def safe_component(value: str) -> str:
    """Keep ordinary names unchanged; make exceptional basenames Windows-safe."""
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    if not value or value in {".", ".."}:
        value = "source"
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
    }
    if value.split(".", 1)[0].upper() in reserved:
        value = "_" + value
    return value


def validate_output_path(path: Path, *, project_root: Path | None = None) -> Path:
    """Reject traversal and symlinks escaping the resolved output root."""
    output_root = (project_root_path(project_root) / "output").resolve()
    resolved = path.resolve()
    if resolved == output_root or not resolved.is_relative_to(output_root):
        raise ValueError(f"Unsafe output path outside output/: {path}")
    for parent in resolved.parents:
        if path_exists(parent) and not parent.is_dir():
            raise ValueError(f"Output parent is not a directory: {parent}")
        if parent == output_root:
            break
    return path


def base_output_directory(target: Path, *, project_root: Path | None = None) -> Path:
    root = project_root_path(project_root)
    target = target.resolve()
    try:
        relative = target.relative_to((root / "input").resolve())
        components = [safe_component(part) for part in relative.parent.parts]
    except ValueError:
        components = []
    return root.joinpath("output", *components, safe_component(target.stem))


def source_identity(target: Path) -> str:
    return os.path.normcase(str(target.resolve()))


def output_directory_key(path: Path, root: Path) -> str:
    validate_output_path(path, project_root=root)
    # Case-fold on every OS to catch Windows collisions in offline Linux tests.
    return path.resolve().relative_to((root / "output").resolve()).as_posix().casefold()


def read_source_owner(directory: Path, root: Path) -> str | None:
    marker = validate_output_path(directory / SOURCE_MARKER, project_root=root)
    if not path_exists(marker):
        return None
    if not marker.is_file():
        raise ValueError(f"Output source marker is not a regular file: {marker}")
    return marker.read_text(encoding="utf-8").removesuffix("\n")


def allocate_output_directory(
    target: Path, root: Path, reserved: set[str], used: dict[str, Path],
) -> Path:
    """Choose a non-colliding base, extension suffix, then stable hash suffix.

    A tiny per-directory source marker preserves ownership across later runs,
    even if a colliding source is absent from the next batch. Never overwrite
    another source's ownership, including when --overwrite is requested.
    """
    base = base_output_directory(target, project_root=root)
    identity = source_identity(target)
    extension = safe_component(target.suffix.lstrip(".") or "source")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    attempt = 0
    while True:
        if attempt == 0:
            candidate = base
        elif attempt == 1:
            candidate = base.with_name(f"{base.name}_{extension}")
        else:
            suffix = "" if attempt == 2 else f"_{attempt - 1}"
            candidate = base.with_name(f"{base.name}_{extension}_{digest}{suffix}")
        key = output_directory_key(candidate, root)
        if path_exists(candidate) and not candidate.is_dir():
            raise ValueError(f"Output directory target is not a directory: {candidate}")
        owner = read_source_owner(candidate, root)
        if (key not in used and (owner is None or owner == identity)
                and (attempt == 0 or key not in reserved or owner == identity)):
            used[key] = target
            return candidate
        attempt += 1


def output_path_for(
    target: Path, kind: str, model_size: str, *, output_dir: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    """Keep existing filenames, changing only their safe parent directory."""
    model_size = validate_model_size(model_size)
    directory = base_output_directory(target, project_root=project_root) if output_dir is None else output_dir
    validate_output_path(directory, project_root=project_root)
    stem = safe_component(target.stem)
    if kind in {"timecode", "simple"}:
        name = f"{kind}-{stem}--{model_size}.txt"
    elif kind == "srt":
        name = f"{stem}--{model_size}.srt"
    else:
        raise ValueError(f"Unknown output kind: {kind}")
    return validate_output_path(directory / name, project_root=project_root)


def expected_output_paths(target: Path, directory: Path, config: dict, root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for enabled, kind in zip(resolve_outputs(config["output_mode"]), ("timecode", "simple", "srt")):
        if enabled:
            paths[kind] = output_path_for(
                target, kind, config["model_size"], output_dir=directory, project_root=root,
            )
    if "timecode" in paths and config["auto_preprocess_block_seconds"] > 0:
        path = paths["timecode"]
        paths["blocks"] = validate_output_path(
            path.with_name(path.stem + ".blocks.txt"), project_root=root,
        )
    return paths


@dataclass
class PlannedInput:
    requested: str
    target: Path | None
    status: str = "runnable"
    reason: str = ""
    output_dir: Path | None = None
    outputs: dict[str, Path] = field(default_factory=dict)


@dataclass
class PreflightPlan:
    items: list[PlannedInput] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duplicates: int = 0

    @property
    def runnable(self) -> list[PlannedInput]:
        return [item for item in self.items if item.status == "runnable"]

    @property
    def skipped_existing(self) -> list[PlannedInput]:
        return [item for item in self.items if item.status == "skipped_existing"]

    @property
    def invalid_or_missing(self) -> list[PlannedInput]:
        return [item for item in self.items if item.status == "invalid_or_missing"]


def first_existing(paths: dict[str, Path]) -> Path | None:
    return next((path for path in paths.values() if path_exists(path)), None)


def build_preflight(
    config: dict, *, overwrite: bool = False, project_root: Path | None = None,
) -> PreflightPlan:
    """Inspect every input and expected output before importing any ML library."""
    root = project_root_path(project_root)
    plan = PreflightPlan()
    seen: set[Path] = set()
    for requested in config["files"]:
        try:
            target = resolve_media_path(requested, project_root=root, warn=plan.warnings.append)
        except (OSError, ValueError, RuntimeError) as error:
            plan.items.append(PlannedInput(
                requested, None, status="invalid_or_missing", reason=error_detail(error, config),
            ))
            continue
        if target in seen:
            plan.duplicates += 1
            plan.warnings.append(f"Duplicate input '{requested}' resolves to {target}; processing it only once.")
            continue
        seen.add(target)
        item = PlannedInput(requested, target)
        try:
            if not target.exists():
                raise FileNotFoundError(f"Input not found: {target}")
            if not target.is_file():
                raise ValueError(f"Input is not a regular file: {target}")
            with target.open("rb") as handle:
                handle.read(1)
        except (OSError, ValueError) as error:
            item.status = "invalid_or_missing"
            item.reason = error_detail(error, config)
        plan.items.append(item)

    reserved: set[str] = set()
    assignable: list[PlannedInput] = []
    for item in plan.items:
        if item.target is None:
            continue
        try:
            reserved.add(output_directory_key(base_output_directory(item.target, project_root=root), root))
            assignable.append(item)
        except (OSError, ValueError, RuntimeError) as error:
            item.status = "invalid_or_missing"
            item.reason = error_detail(error, config)

    input_root = (root / "input").resolve()
    used: dict[str, Path] = {}
    for item in sorted(assignable, key=lambda entry: (
        not entry.target.is_relative_to(input_root), source_identity(entry.target).casefold(),
        source_identity(entry.target),
    )):
        try:
            item.output_dir = allocate_output_directory(item.target, root, reserved, used)
            item.outputs = expected_output_paths(item.target, item.output_dir, config, root)
            base = base_output_directory(item.target, project_root=root)
            if item.output_dir != base:
                plan.warnings.append(f"Output collision for {item.target}: using {item.output_dir}.")
            if (item.output_dir.is_dir() and read_source_owner(item.output_dir, root) is None
                    and any(item.output_dir.iterdir())):
                plan.warnings.append(
                    f"Existing output directory {item.output_dir} has no source marker; "
                    "historical ownership is unknown. Inspect it before --overwrite or "
                    "using a different source/model. Existing-output protection still applies."
                )
            if item.status == "runnable" and not overwrite:
                existing = first_existing(item.outputs)
                if existing is not None:
                    item.status = "skipped_existing"
                    item.reason = f"Existing output: {existing}. Use --overwrite to replace this source's outputs."
        except (OSError, ValueError, RuntimeError) as error:
            item.status = "invalid_or_missing"
            item.reason = error_detail(error, config)
    return plan


def display_preflight(plan: PreflightPlan, *, detailed: bool = False) -> None:
    for message in plan.warnings:
        warning(message)
    for item in plan.items:
        if detailed or item.status != "runnable":
            console.print(Text(f"{item.status}: {item.requested} -> {item.target or '(unresolved)'}"))
            if item.reason:
                console.print(Text("  " + item.reason, style="yellow" if item.status == "skipped_existing" else "red"))
            if detailed:
                for kind, path in item.outputs.items():
                    console.print(Text(f"  {kind}: {path}"))
    console.print(Text(
        f"Preflight: runnable={len(plan.runnable)}, skipped_existing={len(plan.skipped_existing)}, "
        f"invalid_or_missing={len(plan.invalid_or_missing)}, duplicates={plan.duplicates}"
    ))


def claim_output_directory(item: PlannedInput, root: Path) -> None:
    """Record source ownership without changing any transcript or foreign marker."""
    directory = validate_output_path(item.output_dir, project_root=root)
    directory.mkdir(parents=True, exist_ok=True)
    validate_output_path(directory, project_root=root)
    identity = source_identity(item.target)
    owner = read_source_owner(directory, root)
    if owner is not None:
        if owner != identity:
            raise FileExistsError(f"Output directory now belongs to a different source: {directory}")
        return
    marker = validate_output_path(directory / SOURCE_MARKER, project_root=root)
    try:
        with atomic_text_writer(marker) as handle:
            handle.write(identity + "\n")
    except FileExistsError:
        if read_source_owner(directory, root) != identity:
            raise


def flush_vram() -> None:
    """Release model references before calling; run even after a stage fails."""
    gc.collect()
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError as error:
            warning(f"CUDA cache cleanup failed: {error_detail(error)}")


def transcribe_file(target: Path, config: dict) -> dict:
    """Keep the per-file ASR -> Alignment -> optional Diarization pipeline."""
    load_torch()
    wx = load_whisperx()
    model_size = config.get("model_size", "large-v3-turbo")
    device = config.get("device", "cuda")
    language_cfg = config.get("language", "uk")
    language = None if language_cfg.lower() == "auto" else language_cfg
    do_diarize = config.get("diarize", False)
    hf_token = config.get("hf_token", "") or None
    min_speakers = config.get("min_speakers", 0)
    max_speakers = config.get("max_speakers", 0)
    compute_type = "float16" if device == "cuda" else "int8"

    # The language-specific hint is still the only filler-removal mechanism.
    initial_prompt = (config.get("initial_prompt") or None) if config.get("remove_fillers") else None
    asr_options = {"beam_size": 5}
    if initial_prompt:
        asr_options["initial_prompt"] = initial_prompt

    # ── Stage 1 — ASR ──────────────────────────────────────────────────
    model = None
    try:
        with console.status(f"[bold yellow]1/3 Loading WhisperX model '{model_size}'...", spinner="dots"):
            model = wx.load_model(
                model_size, device, compute_type=compute_type,
                language=language, asr_options=asr_options,
            )
        audio = wx.load_audio(str(target))
        with console.status("[bold blue]1/3 Transcribing...", spinner="arc"):
            result = model.transcribe(audio, batch_size=config.get("batch_size", 8), language=language)
    finally:
        model = None
        flush_vram()

    detected_lang = result.get("language") or language or "uk"

    # ── Stage 2 — Alignment ────────────────────────────────────────────
    align_model = None
    align_metadata = None
    try:
        with console.status(f"[bold cyan]2/3 Aligning words ({detected_lang})...", spinner="arc"):
            align_model, align_metadata = wx.load_align_model(language_code=detected_lang, device=device)
            aligned = wx.align(
                result["segments"], align_model, align_metadata, audio, device,
                return_char_alignments=False,
            )
            result["segments"] = aligned["segments"]
    except Exception as error:
        # Optional stage: retain raw ASR timings when an aligner is unavailable.
        warning(f"alignment skipped ({error_detail(error, config)}).")
    finally:
        align_model = None
        align_metadata = None
        flush_vram()

    # ── Stage 3 — Diarization ──────────────────────────────────────────
    if do_diarize and hf_token:
        diarize_model = None
        try:
            with console.status("[bold magenta]3/3 Diarizing speakers (pyannote)...", spinner="bouncingBar"):
                # WhisperX 3.8.x: pyannote v4, community-1, and token=.
                from whisperx.diarize import DiarizationPipeline

                diarize_model = DiarizationPipeline(token=hf_token, device=device)
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=min_speakers if min_speakers > 0 else None,
                    max_speakers=max_speakers if max_speakers > 0 else None,
                )
                result = wx.assign_word_speakers(diarize_segments, result)
        except Exception as error:
            warning(
                f"Diarization failed: {error_detail(error, config)}. Check the pyannote "
                "license agreement and HF token; continuing without speaker tags."
            )
        finally:
            diarize_model = None
            flush_vram()

    last_end = result["segments"][-1].get("end", 0.0) if result.get("segments") else 0.0
    console.print(Text(f"Duration: {format_timestamp(last_end)} | Language: {detected_lang}\n", style="dim"))
    return result


def write_outputs(
    result: dict, target: Path, model_size: str,
    write_timecoded: bool, write_simple: bool, write_srt: bool,
    *, output_dir: Path | None = None, overwrite: bool = False,
) -> Path | None:
    """Write existing formats using per-file atomic replacement, not a transaction."""
    paths = {
        kind: output_path_for(target, kind, model_size, output_dir=output_dir)
        for enabled, kind in zip(
            (write_timecoded, write_simple, write_srt), ("timecode", "simple", "srt"),
        ) if enabled
    }
    with ExitStack() as stack:
        handles = {
            kind: stack.enter_context(atomic_text_writer(path, overwrite=overwrite))
            for kind, path in paths.items()
        }
        for index, segment in enumerate(result["segments"], start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            speaker_tag = f"[{segment['speaker']}] " if segment.get("speaker") else ""
            body = f"{speaker_tag}{text}"
            if write_timecoded:
                console.print(
                    Text(f"[{start} -> {end}]", style="bold yellow"),
                    Text(speaker_tag, style="bold magenta") if speaker_tag else Text(""),
                    Text(text),
                )
            else:
                console.print(Text(f"• {body}", style="green"))
            if "timecode" in handles:
                handles["timecode"].write(f"[{start} -> {end}] {body}\n")
            if "simple" in handles:
                handles["simple"].write(f"{body}\n")
            if "srt" in handles:
                handles["srt"].write(
                    f"{index}\n"
                    f"{format_timestamp(segment['start'], srt_format=True)} --> "
                    f"{format_timestamp(segment['end'], srt_format=True)}\n"
                    f"{body}\n\n"
                )
    return paths.get("timecode")


def run_preprocess(
    timecoded_files: list[Path], block_seconds: float, *, overwrite: bool = False,
) -> int:
    """Return the number of failed sources so the batch cannot report false success."""
    console.rule("[bold cyan]Preprocessing timecoded outputs[/bold cyan]")
    try:
        import preprocess
    except ImportError as error:
        warning(f"preprocess.py not importable: {error_detail(error)}")
        return len(timecoded_files)
    failures = 0
    for path in timecoded_files:
        try:
            validate_output_path(path)
            validate_output_path(preprocess.blocks_path_for(path))
            out = preprocess.process_file(path, block_seconds, outdir=None, overwrite=overwrite)
            size_kb = out.stat().st_size / 1024
            console.print(Text(f"✓ {path.name} → {out.name} ({size_kb:.1f} KB)", style="green"))
        except Exception as error:
            console.print(Text(f"✕ Preprocessing {path}: {error_detail(error)}", style="red"))
            failures += 1
    return failures


@dataclass
class BatchCounts:
    succeeded: int = 0
    skipped_existing: int = 0
    missing_or_invalid: int = 0
    failed_transcription: int = 0
    failed_write: int = 0
    failed_preprocess: int = 0

    @property
    def failures(self) -> int:
        return self.missing_or_invalid + self.failed_transcription + self.failed_write + self.failed_preprocess


def display_summary(counts: BatchCounts) -> None:
    table = Table(show_header=False, box=None)
    for name, value in vars(counts).items():
        table.add_row(name, str(value))
    success = counts.failures == 0
    console.print(Panel(
        table, title="All tasks completed" if success else "Batch partial or failed",
        border_style="bright_green" if success else "red",
    ))


def cuda_status() -> str:
    """Diagnostic only: no model loads and no failure solely for a missing GPU."""
    try:
        runtime = load_torch()
        available = runtime.cuda.is_available()
        version = runtime.version.cuda
        return f"CUDA runtime: {version or 'CPU-only torch'}; GPU available: {'yes' if available else 'no'}"
    except (ImportError, OSError, RuntimeError) as error:
        return f"CUDA status unavailable: {error_detail(error)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WhisperBlocks: safe local batch transcription")
    parser.add_argument("files", nargs="*", help="Media paths overriding the config's files list")
    parser.add_argument("--config", type=Path, help="Config path (relative to project root or absolute)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs for targeted sources")
    parser.add_argument("--check-config", action="store_true", help="Validate and report CUDA/input/output status; no models")
    parser.add_argument("--dry-run", action="store_true", help="Show resolved input/output plan; no models")
    args = parser.parse_intermixed_args(argv)
    root = project_root_path()
    try:
        (root / "input").mkdir(parents=True, exist_ok=True)
        (root / "output").mkdir(parents=True, exist_ok=True)
        config_path = CONFIG_PATH if args.config is None else args.config.expanduser()
        if not config_path.is_absolute():
            config_path = root / config_path
        if not ensure_config(config_path, seed=args.config is None):
            return 0
        with config_path.open("rb") as handle:
            raw_config = tomllib.load(handle)
        config = validate_config(raw_config, args.files)
    except (OSError, ValueError, RuntimeError) as error:
        console.print(Panel(Text(f"Configuration error: {error_detail(error)}"), border_style="red"))
        return 2

    for message in configuration_warnings(config):
        warning(message)
    plan = build_preflight(config, overwrite=args.overwrite, project_root=root)
    display_dashboard(config, config["files"])
    display_preflight(plan, detailed=args.dry_run or args.check_config)
    if args.check_config or args.dry_run:
        if args.check_config:
            console.print(Text(cuda_status()))
        console.print(Text("Preflight only — no transcription models loaded and no outputs written."))
        return 1 if plan.invalid_or_missing else 0

    counts = BatchCounts(
        skipped_existing=len(plan.skipped_existing), missing_or_invalid=len(plan.invalid_or_missing),
    )
    runnable = plan.runnable
    if not runnable:
        display_summary(counts)
        return 1 if counts.failures else 0

    # Only a real, non-empty transcription queue may fail on CUDA availability.
    try:
        runtime = load_torch()
        if config["device"] == "cuda" and not runtime.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Check the CUDA-enabled torch wheel "
                "(torch.version.cuda must not be None), NVIDIA GPU and driver. "
                "Run install.bat to repair the CUDA PyTorch install, or set device = \"cpu\" "
                "in your config for CPU fallback."
            )
    except (ImportError, OSError, RuntimeError) as error:
        console.print(Panel(Text(f"Runtime error: {error_detail(error, config)}"), border_style="red"))
        counts.failed_transcription = len(runnable)
        display_summary(counts)
        return 1

    generated_timecoded: list[Path] = []
    for index, item in enumerate(runnable, start=1):
        console.rule(Text(f"File [{index}/{len(runnable)}]: {item.target.name}", style="bold cyan"))
        started = time.monotonic()
        try:
            # Recheck after preflight in case outputs appeared while processing
            # an earlier source. The atomic writer checks again before commit.
            for path in item.outputs.values():
                validate_output_path(path)
            existing = first_existing(item.outputs)
            if existing is not None and not args.overwrite:
                counts.skipped_existing += 1
                warning(f"Skipped {item.target}: output appeared at {existing}; use --overwrite to replace.")
                continue
            claim_output_directory(item, root)
        except (OSError, ValueError, RuntimeError) as error:
            counts.failed_write += 1
            console.print(Text(f"✕ Cannot prepare outputs for {item.target}: {error_detail(error, config)}", style="red"))
            continue
        transcription_failed = False
        try:
            result = transcribe_file(item.target, config)
        except Exception as error:
            counts.failed_transcription += 1
            console.print(Text(f"✕ Transcription {item.target}: {error_detail(error, config)}", style="red"))
            transcription_failed = True
        if transcription_failed:
            # The handler has ended: its exception/traceback no longer retains
            # model-method frames. Flush again before the next batch source.
            flush_vram()
            continue
        try:
            timecoded = write_outputs(
                result, item.target, config["model_size"], *resolve_outputs(config["output_mode"]),
                output_dir=item.output_dir, overwrite=args.overwrite,
            )
        except Exception as error:
            counts.failed_write += 1
            console.print(Text(f"✕ Writing {item.target}: {error_detail(error, config)}", style="red"))
            continue
        counts.succeeded += 1
        if timecoded is not None:
            generated_timecoded.append(timecoded)
        console.print(Text(f"✓ Transcribed and written in {time.monotonic() - started:.1f}s\n", style="green"))

    if config["auto_preprocess_block_seconds"] > 0 and generated_timecoded:
        failures = run_preprocess(
            generated_timecoded, config["auto_preprocess_block_seconds"], overwrite=args.overwrite,
        )
        counts.failed_preprocess += failures
        counts.succeeded -= failures
    display_summary(counts)
    return 1 if counts.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
