import hashlib
from pathlib import Path

import pytest

from src.domain.services.file_fingerprint import compute_file_sha256


def test_compute_file_sha256_matches_full_file_digest(tmp_path: Path) -> None:
    payload = b"binary-pdf-payload" * 1000
    source = tmp_path / "manual.pdf"
    source.write_bytes(payload)

    assert compute_file_sha256(source) == hashlib.sha256(payload).hexdigest()


def test_compute_file_sha256_is_chunk_size_independent(tmp_path: Path) -> None:
    source = tmp_path / "manual.pdf"
    source.write_bytes(b"abcdefghij" * 100)

    assert compute_file_sha256(source, block_size=3) == compute_file_sha256(source, block_size=1024)


def test_compute_file_sha256_rejects_invalid_block_size(tmp_path: Path) -> None:
    source = tmp_path / "manual.pdf"
    source.write_bytes(b"payload")

    with pytest.raises(ValueError):
        compute_file_sha256(source, block_size=0)
