#!/usr/bin/env python3
"""
preprocess.py — Merge raw Whisper transcript segments into fixed-duration
blocks for downstream AI processing.

Input (timecode-*.txt, whole-second segment timestamps):
    [00:02 -> 00:14] Some text that may
    wrap onto a second line.
    [00:14 -> 00:22] [SPEAKER_00] Next segment...

Output:
    Source: timecode-lecture--large-v3-turbo.txt
    Segments: 2 | Blocks: 1 (~60s each)

    === BLOCK 01 | 00:02 → 00:22 ===
    Some text that may wrap onto a second line. [SPEAKER_00] Next segment...

An optional pre-segment header is passed through unchanged if supplied by
another producer. transcript.py does not write a duration/language header.

Usage (from the project root):
    python src/preprocess.py output/lecture/timecode-lecture--large-v3-turbo.txt
    python src/preprocess.py "output/*/timecode-*.txt" --block 90
    python src/preprocess.py "output/*/timecode-*.txt" --overwrite
"""

import argparse
import glob
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from file_io import atomic_text_writer, path_exists


SEGMENT_RE = re.compile(r"^\[([0-9:]+)\s*->\s*([0-9:]+)\]\s*(.*)$")
ARROW_HEADER_RE = re.compile(r"^\[[^\]]*->[^\]]*\]")


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Block:
    index: int
    start: float
    end: float
    text: str
    segment_count: int


def validate_block_seconds(value: float) -> float:
    """Validate programmatic calls as well as the standalone CLI."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("block duration must be a finite number greater than 0")
    try:
        valid = math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("block duration must be a finite number greater than 0")
    return float(value)


def parse_timestamp(s: str) -> float:
    """Parse non-negative MM:SS or HH:MM:SS; minutes/seconds must be < 60."""
    parts = s.split(":")
    if len(parts) not in (2, 3) or any(not re.fullmatch(r"[0-9]+", p) for p in parts):
        raise ValueError(f"Invalid timestamp '{s}' (expected MM:SS or HH:MM:SS)")
    try:
        nums = [int(p) for p in parts]
        if nums[-1] >= 60 or nums[-2] >= 60:
            raise ValueError("minutes and seconds must be less than 60")
        seconds = nums[0] * 60 + nums[1] if len(nums) == 2 else (
            nums[0] * 3600 + nums[1] * 60 + nums[2]
        )
        value = float(seconds)
        if not math.isfinite(value):
            raise ValueError("timestamp must be finite")
        return value
    except (ValueError, OverflowError) as error:
        raise ValueError(f"Invalid timestamp '{s}': {error}") from error


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    sec = int(seconds)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_transcript(text: str) -> tuple[str, list[Segment]]:
    """Parse segments and optional header, preserving ordinary wrapped text.

    Starts must be chronological; overlapping segment ranges are legitimate.
    Only a bracketed arrow header is treated as a malformed timestamp, not
    ordinary [SPEAKER_00] / [inaudible] / other bracketed continuation text.
    """
    header_lines: list[str] = []
    segments: list[Segment] = []
    current: Segment | None = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip()
        if not line.strip():
            continue
        match = SEGMENT_RE.match(line)
        if match:
            try:
                start = parse_timestamp(match.group(1))
                end = parse_timestamp(match.group(2))
                if end < start:
                    raise ValueError("segment end is before its start")
                if current is not None and start < current.start:
                    raise ValueError("segment starts must be non-decreasing")
            except ValueError as error:
                raise ValueError(f"Line {line_number}: {error}") from error
            if current is not None:
                segments.append(current)
            current = Segment(start=start, end=end, text=match.group(3).strip())
        elif ARROW_HEADER_RE.match(line):
            raise ValueError(f"Line {line_number}: malformed timestamp header: {line}")
        elif current is None:
            header_lines.append(line)
        else:
            current.text = (current.text + " " + line.strip()).strip()

    if current is not None:
        segments.append(current)
    if not segments:
        raise ValueError("No timestamped segments found")
    return "\n".join(header_lines), segments


def merge_into_blocks(segments: list[Segment], block_seconds: float) -> list[Block]:
    """Greedily merge consecutive segments into blocks of ~block_seconds."""
    block_seconds = validate_block_seconds(block_seconds)
    blocks: list[Block] = []
    if not segments:
        return blocks

    buf: list[Segment] = []
    block_start = segments[0].start

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = " ".join(seg.text for seg in buf).strip()
        text = re.sub(r"\s{2,}", " ", text)
        blocks.append(Block(
            index=len(blocks) + 1,
            start=buf[0].start,
            end=buf[-1].end,
            text=text,
            segment_count=len(buf),
        ))
        buf = []

    for seg in segments:
        if buf and (seg.end - block_start) > block_seconds:
            flush()
            block_start = seg.start
        buf.append(seg)

    flush()
    return blocks


def render(header: str, blocks: list[Block], source: Path, block_seconds: float) -> str:
    """Produce the existing block text format, including any supplied header."""
    lines = [f"Source: {source.name}"]
    if header:
        lines.append(header)
    total_segments = sum(b.segment_count for b in blocks)
    lines.append(
        f"Segments: {total_segments} | "
        f"Blocks: {len(blocks)} (~{int(block_seconds)}s each)"
    )
    lines.append("")
    for block in blocks:
        lines.append(
            f"=== BLOCK {block.index:02d} | "
            f"{format_timestamp(block.start)} → {format_timestamp(block.end)} ==="
        )
        lines.append(block.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def blocks_path_for(path: Path, outdir: Path | None = None) -> Path:
    return (path.parent if outdir is None else outdir) / (path.stem + ".blocks.txt")


def process_file(
    path: Path, block_seconds: float, outdir: Path | None = None,
    *, overwrite: bool = False,
) -> Path:
    block_seconds = validate_block_seconds(block_seconds)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")
    if not path.is_file():
        raise ValueError(f"Transcript is not a regular file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError as error:
        raise ValueError(f"Transcript is not readable UTF-8 text: {path}") from error
    header, segments = parse_transcript(text)
    blocks = merge_into_blocks(segments, block_seconds)
    output = render(header, blocks, path, block_seconds)
    out_path = blocks_path_for(path, outdir)
    if out_path.resolve() == path.resolve():
        raise ValueError(f"Output would replace its own input: {path}")
    with atomic_text_writer(out_path, overwrite=overwrite) as handle:
        handle.write(output)
    return out_path


def expand_inputs(patterns: list[str]) -> tuple[list[Path], list[str]]:
    """Expand Windows CLI globs, excluding generated blocks only for globs."""
    paths: list[Path] = []
    errors: list[str] = []
    seen: set[Path] = set()
    for original in patterns:
        pattern = original.replace("\\", os.sep) if os.name != "nt" else original
        # An existing literal filename can contain brackets; do not interpret
        # it as a character-class pattern unless literal lookup failed.
        wildcard = glob.has_magic(pattern) and not path_exists(Path(pattern))
        matches = sorted(glob.glob(pattern, recursive=True)) if wildcard else (
            [pattern] if path_exists(Path(pattern)) else []
        )
        if wildcard:
            matches = [p for p in matches if not p.lower().endswith(".blocks.txt")]
        if not matches:
            errors.append(f"No input transcripts matched: {original}")
            continue
        for match in matches:
            path = Path(match)
            try:
                resolved = path.resolve()
            except (OSError, RuntimeError) as error:
                errors.append(f"Cannot resolve {path}: {error}")
                continue
            if resolved not in seen:
                seen.add(resolved)
                paths.append(path)
    return paths, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge raw Whisper transcript segments into fixed-duration blocks."
    )
    parser.add_argument("files", nargs="+", help="Transcript paths or glob patterns")
    parser.add_argument("--block", "-b", type=float, default=60.0,
                        help="Target block duration in seconds (default: 60)")
    parser.add_argument("--outdir", "-o", type=Path, default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing block files")
    args = parser.parse_intermixed_args(argv)
    try:
        validate_block_seconds(args.block)
    except ValueError as error:
        parser.error(str(error))

    paths, errors = expand_inputs(args.files)
    for error in errors:
        print(f"✕ {error}", file=sys.stderr)
    exit_code = 1 if errors else 0
    for path in paths:
        out_path = blocks_path_for(path, args.outdir)
        try:
            out = process_file(path, args.block, args.outdir, overwrite=args.overwrite)
            size_kb = out.stat().st_size / 1024
        except FileExistsError as error:
            if args.overwrite:
                print(f"✕ {path}: {error}", file=sys.stderr)
                exit_code = 1
            else:
                print(f"Skipped existing: {out_path}. Use --overwrite to replace it.")
            continue
        except (OSError, ValueError) as error:
            print(f"✕ {path}: {error}", file=sys.stderr)
            exit_code = 1
            continue
        print(f"✓ {path.name} → {out.name} ({size_kb:.1f} KB)")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
