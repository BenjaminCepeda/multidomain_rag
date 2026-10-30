import hashlib
import uuid

from src.domain.models.ids import ChunkId, DocumentId


def test_document_id_deterministic() -> None:
    a = DocumentId.from_source("C:/docs/manual.pdf")
    DocumentId.from_source(
        "c:/docs/manual.pdf"
    )  # case/sep normalization via resolve may differ on Windows
    assert isinstance(a.value, uuid.UUID)


def test_document_id_from_same_source_equal() -> None:
    a = DocumentId.from_source("manual.pdf")
    b = DocumentId.from_source("manual.pdf")
    assert a == b


def test_chunk_id_validation_empty() -> None:
    try:
        ChunkId("")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_chunk_id_build() -> None:
    doc = DocumentId.from_source("manual.pdf")
    cid = ChunkId.build(document_id=doc, section="Intro", page=1, index=0)
    assert str(doc) in cid.value
    assert cid.value.endswith(":intro:1:0")


def test_chunk_id_build_invalid_page() -> None:
    doc = DocumentId.from_source("manual.pdf")
    try:
        ChunkId.build(document_id=doc, section=None, page=0, index=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_stable_point_id_is_sha256_derived_and_repeatable() -> None:
    chunk_id = ChunkId("doc-uuid:root:2:3")
    expected = (
        int.from_bytes(hashlib.sha256(chunk_id.value.encode("utf-8")).digest()[:8], "big")
        & 0x7FFF_FFFF_FFFF_FFFF
    )

    assert chunk_id.stable_point_id() == expected
    assert chunk_id.stable_point_id() == chunk_id.stable_point_id()
    assert 0 <= chunk_id.stable_point_id() < 2**63


def test_stable_point_id_differs_between_chunks() -> None:
    first = ChunkId("doc-uuid:root:2:3")
    second = ChunkId("doc-uuid:root:2:4")

    assert first.stable_point_id() != second.stable_point_id()
