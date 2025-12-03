#!/usr/bin/env python3
"""
Shared helpers for staged book cleaning.
"""
import os
from pathlib import Path
from typing import Iterable, Iterator, Set, Tuple


def is_binary_file(path: Path, blocksize: int = 1024) -> bool:
    """Heuristically detect binary files."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(blocksize)
    except OSError:
        return True
    if not chunk:
        return False
    if b"\x00" in chunk:
        return True
    text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
    nontext = sum(b not in text_chars for b in chunk)
    return (nontext / len(chunk)) > 0.30


def auto_read_text(path: Path) -> str:
    """UTF-8 with latin-1 fallback, ignoring undecodable characters."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")


def iter_txt_files(roots: Iterable[Path]) -> Iterator[Path]:
    """Yield unique *.txt files across roots, de-duplicating inodes."""
    seen: Set[Tuple[int, int]] = set()
    for root in roots:
        for path in sorted(root.rglob("*.txt")):
            # Skip macOS resource forks / metadata files.
            if path.name.startswith("._") or "__MACOSX" in path.parts:
                continue
            if not path.is_file():
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            yield path


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
