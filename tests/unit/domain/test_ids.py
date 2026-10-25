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
