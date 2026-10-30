"""Stable fingerprints for physical source documents."""

from __future__ import annotations

import hashlib
from pathlib import Path

_READ_BLOCK_SIZE = 1024 * 1024


def compute_file_sha256(path: str | Path, block_size: int = _READ_BLOCK_SIZE) -> str:
    """Return the SHA-256 digest of a file's complete binary contents.

    The digest is computed incrementally so large manuals do not need to be
    loaded into memory. The result is deterministic across processes and is
    safe to use as a deduplication key.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()
