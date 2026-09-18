#!/usr/bin/env python3
"""
preprocess.py — Merge raw Whisper transcript segments into fixed-duration
blocks for downstream AI processing.

Input (raw Whisper output):
    Duration: 30:46 | Language: uk (100.0%)

    [00:02 -> 00:14] Some text that may
    wrap onto a second line.
    [00:14 -> 00:22] Next segment...

Output:
    Source: zvvb_pz1_meet.wav
    Duration: 30:46 | Language: uk (100.0%)
    Segments: 187 | Blocks: 31 (~60s each)

    === BLOCK 01 | 00:02 → 01:08 ===
    Merged text of the block, no timestamps inside.

    === BLOCK 02 | 01:08 → 02:33 ===
    ...

Usage:
    python preprocess.py transcript.txt
    python preprocess.py transcript.txt --block 90
    python preprocess.py *.txt --outdir blocks/
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


SEGMENT_RE = re.compile(r"^\[([\d:]+)\s*->\s*([\d:]+)\]\s*(.*)$")


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


def parse_timestamp(s: str) -> float:
    """Parse 'MM:SS' or 'HH:MM:SS' into seconds."""
    parts = s.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"Invalid timestamp '{s}'") from e
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    raise ValueError(f"Invalid timestamp '{s}'")


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    sec = int(seconds)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_transcript(text: str) -> tuple[str, list[Segment]]:
    """
    Parse raw Whisper output into (header, segments).

    Continuation lines (not starting with a timestamp) are appended to the
    previous segment's text with a single space.
    """
    header_lines: list[str] = []
    segments: list[Segment] = []
    current: Segment | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        m = SEGMENT_RE.match(line)
        if m:
            if current is not None:
                segments.append(current)
            start_s, end_s, body = m.group(1), m.group(2), m.group(3)
            current = Segment(
                start=parse_timestamp(start_s),
                end=parse_timestamp(end_s),
                text=body.strip(),
            )
        else:
            if current is None:
                # Pre-segment content: "Duration: ... | Language: ..."
                header_lines.append(line)
            else:
                # Continuation of the current segment
                current.text = (current.text + " " + line.strip()).strip()

    if current is not None:
        segments.append(current)

    return "\n".join(header_lines), segments


def merge_into_blocks(segments: list[Segment], block_seconds: float) -> list[Block]:
    """Greedily merge consecutive segments into blocks of ~block_seconds."""
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
        # Collapse double spaces that can appear at segment boundaries
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
        # Would adding this segment push the block past the limit?
        if buf and (seg.end - block_start) > block_seconds:
            flush()
            block_start = seg.start
        buf.append(seg)

    flush()
    return blocks


def render(header: str, blocks: list[Block], source: Path, block_seconds: float) -> str:
    """Produce the output text."""
    lines: list[str] = []

    lines.append(f"Source: {source.name}")
    if header:
        lines.append(header)
    total_segments = sum(b.segment_count for b in blocks)
    lines.append(
        f"Segments: {total_segments} | "
        f"Blocks: {len(blocks)} (~{int(block_seconds)}s each)"
    )
    lines.append("")

    for b in blocks:
        lines.append(
            f"=== BLOCK {b.index:02d} | "
            f"{format_timestamp(b.start)} → {format_timestamp(b.end)} ==="
        )
        lines.append(b.text)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def process_file(path: Path, block_seconds: float, outdir: Path | None) -> Path:
    text = path.read_text(encoding="utf-8")
    header, segments = parse_transcript(text)

    if not segments:
        raise ValueError(f"No timestamped segments found in {path.name}")

    blocks = merge_into_blocks(segments, block_seconds)
    output = render(header, blocks, path, block_seconds)

    if outdir is None:
        out_path = path.with_name(path.stem + ".blocks.txt")
    else:
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / (path.stem + ".blocks.txt")

    out_path.write_text(output, encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge raw Whisper transcript segments into fixed-duration blocks."
    )
    parser.add_argument("files", nargs="+", type=Path,
                        help="One or more raw transcript .txt files")
    parser.add_argument("--block", "-b", type=float, default=60.0,
                        help="Target block duration in seconds (default: 60)")
    parser.add_argument("--outdir", "-o", type=Path, default=None,
                        help="Output directory (default: same as input)")
    args = parser.parse_args()

    exit_code = 0
    for path in args.files:
        if not path.exists():
            print(f"✕ Not found: {path}", file=sys.stderr)
            exit_code = 1
            continue
        try:
            out = process_file(path, args.block, args.outdir)
        except Exception as e:
            print(f"✕ {path.name}: {e}", file=sys.stderr)
            exit_code = 1
            continue
        size_kb = out.stat().st_size / 1024
        print(f"✓ {path.name} → {out.name} ({size_kb:.1f} KB)")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())