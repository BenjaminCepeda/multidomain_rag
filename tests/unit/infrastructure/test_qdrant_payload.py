from src.domain.models.document import ManualChunk
from src.domain.models.ids import ChunkId
from src.infrastructure.adapters.qdrant_adapter import QdrantAdapter

REQUIRED_VISUAL_PAYLOAD_KEYS = {
    "document_name",
    "document_title",
    "document_context_summary",
    "page_number",
    "image_title",
    "theme_summary",
    "surrounding_text_context",
    "image_path",
}


def _visual_chunk() -> ManualChunk:
    return ManualChunk(
        chunk_id=ChunkId("doc-uuid:visual:3:0"),
        content="[IMAGEN] Diagrama de flujo de inferencia en el borde",
        page=3,
        is_visual=True,
        metadata={
            "document_name": "EdgeAI_manual.pdf",
            "document_title": "EdgeAI manual",
            "document_context_summary": "Guia de despliegue de modelos en dispositivos.",
            "page_number": 3,
            "image_title": "Diagrama de flujo",
            "theme_summary": "inferencia, borde, modelo",
            "surrounding_text_context": "El modelo se ejecuta localmente.",
            "image_path": "/data/extracted_images/abc_p3_i0.png",
            "source": "EdgeAI_manual.pdf",
            "image_paths": "/data/extracted_images/abc_p3_i0.png",
            "file_hash": "abc123def456",
        },
    )


def test_visual_payload_exposes_enriched_context_fields() -> None:
    payload = QdrantAdapter._build_payload(_visual_chunk())

    assert REQUIRED_VISUAL_PAYLOAD_KEYS.issubset(payload.keys())
    assert payload["page"] == 3
    assert payload["page_number"] == 3
    assert payload["is_visual"] is True
    assert payload["image_path"] == "/data/extracted_images/abc_p3_i0.png"
    assert payload["image_paths"] == "/data/extracted_images/abc_p3_i0.png"
    assert payload["file_hash"] == "abc123def456"


def test_text_payload_falls_back_to_filename_as_source() -> None:
    chunk = ManualChunk(
        chunk_id=ChunkId("doc-uuid:root:1:0"),
        content="Texto tecnico del manual.",
        page=1,
        metadata={"filename": "manual.pdf", "document_title": "Manual"},
    )

    payload = QdrantAdapter._build_payload(chunk)

    assert payload["source"] == "manual.pdf"
    assert payload["document_title"] == "Manual"


class _FakeCountResult:
    def __init__(self, count: int) -> None:
        self.count = count


class _FakeQdrantClient:
    def __init__(self, count: int) -> None:
        self._count = count
        self.count_calls: list[dict[str, object]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return True

    def count(
        self,
        *,
        collection_name: str,
        count_filter: object = None,
        exact: bool = True,
    ) -> _FakeCountResult:
        self.count_calls.append(
            {
                "collection_name": collection_name,
                "count_filter": count_filter,
                "exact": exact,
            }
        )
        return _FakeCountResult(self._count)


def test_exists_by_file_hash_filters_by_payload_hash() -> None:
    client = _FakeQdrantClient(count=1)
    adapter = object.__new__(QdrantAdapter)
    adapter._client = client
    adapter._collection_name = "technical_docs"
    adapter._vector_dim = 1024

    assert adapter.exists_by_file_hash("abc123") is True

    assert len(client.count_calls) == 1
    call = client.count_calls[0]
    assert call["collection_name"] == "technical_docs"
    count_filter = call["count_filter"]
    condition = count_filter.must[0]  # type: ignore[attr-defined]
    assert condition.key == "file_hash"
    assert condition.match.value == "abc123"  # type: ignore[attr-defined]


def test_exists_by_file_hash_returns_false_for_empty_hash() -> None:
    adapter = object.__new__(QdrantAdapter)

    assert adapter.exists_by_file_hash("") is False
