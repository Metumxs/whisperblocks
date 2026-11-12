"""Small shared helpers for per-file atomic UTF-8 output (not transactions)."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


def path_exists(path: Path) -> bool:
    """Include broken symlinks in overwrite protection."""
    return os.path.lexists(path)


@contextmanager
def atomic_text_writer(path: Path, *, overwrite: bool = False) -> Iterator[TextIO]:
    """Close a temporary sibling before replacing the destination.

    Check for existing destinations both before writing and before committing.
    This protects ordinary sequential runs, but is not a cross-process lock.
    Each destination is committed separately; a group of files is not atomic.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path_exists(path):
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            yield handle
        if not overwrite and path_exists(path):
            raise FileExistsError(
                f"Output appeared while writing: {path}. Use --overwrite to replace it."
            )
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
