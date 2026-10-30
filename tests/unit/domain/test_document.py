from src.domain.models.document import ManualChunk
from src.domain.models.ids import ChunkId


def test_embedding_input_prefers_compact_text() -> None:
    chunk = ManualChunk(
        chunk_id=ChunkId("doc-uuid:visual:1:0"),
        content="Contenido rico y extenso para el payload y el LLM.",
        embedding_text="Texto compacto para el embedding denso.",
        is_visual=True,
    )

    assert chunk.embedding_input == "Texto compacto para el embedding denso."
    assert chunk.content == "Contenido rico y extenso para el payload y el LLM."


def test_embedding_input_falls_back_to_content() -> None:
    chunk = ManualChunk(
        chunk_id=ChunkId("doc-uuid:root:1:0"),
        content="Texto tecnico del manual.",
    )

    assert chunk.embedding_input == "Texto tecnico del manual."


def test_embedding_input_falls_back_when_compact_text_is_blank() -> None:
    chunk = ManualChunk(
        chunk_id=ChunkId("doc-uuid:visual:1:0"),
        content="Contenido rico.",
        embedding_text="   ",
    )

    assert chunk.embedding_input == "Contenido rico."
