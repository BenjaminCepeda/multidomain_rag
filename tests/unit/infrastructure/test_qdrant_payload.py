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
