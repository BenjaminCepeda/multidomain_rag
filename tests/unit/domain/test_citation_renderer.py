from src.domain.models.document import ManualChunk
from src.domain.models.ids import ChunkId
from src.domain.services.citation_renderer import CitationRenderer


def _chunk(
    chunk_id: str,
    page: int,
    *,
    source: str | None = None,
    document_name: str | None = None,
) -> ManualChunk:
    metadata: dict[str, str | int | float | bool] = {"page_number": page}
    if source is not None:
        metadata["source"] = source
    if document_name is not None:
        metadata["document_name"] = document_name
    return ManualChunk(
        chunk_id=ChunkId(chunk_id),
        content="Texto tecnico del fragmento.",
        page=page,
        metadata=metadata,
    )


def test_placeholder_is_replaced_with_the_real_source_for_its_page() -> None:
    renderer = CitationRenderer(
        [
            _chunk("page-3", 3, source="manual_principal.pdf"),
            _chunk("page-7", 7, source="anexo_tecnico.pdf"),
        ]
    )

    rendered, stats = renderer.render(
        "La inferencia ocurre en el dispositivo [Archivo, Pág 7]. "
        "El modelo se entrena antes [Archivo, Pág 3]."
    )

    assert "[anexo_tecnico.pdf, Pág 7]" in rendered
    assert "[manual_principal.pdf, Pág 3]" in rendered
    assert "Archivo" not in rendered
    assert stats.placeholders == 2
    assert stats.replaced == 2
    assert stats.used_fallback_source is False


def test_placeholder_without_page_match_uses_the_top_fragment_source() -> None:
    renderer = CitationRenderer([_chunk("page-3", 3, document_name="manual_principal.pdf")])

    rendered, stats = renderer.render("Ver [Documento, Pág 99].")

    assert rendered == "Ver [manual_principal.pdf, Pág 99]."
    assert stats.replaced == 1
    assert stats.used_fallback_source is True


def test_real_citations_are_preserved_unchanged() -> None:
    renderer = CitationRenderer([_chunk("page-3", 3, source="manual_principal.pdf")])
    original = "Ver [Efficient_Acceleration_Review.pdf, Pág 24] y [manual_principal.pdf, Pág 3]."

    rendered, stats = renderer.render(original)

    assert rendered == original
    assert stats.placeholders == 0
    assert stats.replaced == 0


def test_placeholder_is_left_untouched_when_no_source_is_available() -> None:
    renderer = CitationRenderer([_chunk("page-3", 3)])

    rendered, stats = renderer.render("Ver [Archivo, Pág 3].")

    assert rendered == "Ver [Archivo, Pág 3]."
    assert stats.placeholders == 1
    assert stats.replaced == 0


def test_source_is_reduced_to_the_real_filename() -> None:
    renderer = CitationRenderer([_chunk("page-3", 3, source="/data/raw/manual_principal.pdf")])

    rendered, _stats = renderer.render("Ver [Fuente, página 3].")

    assert rendered == "Ver [manual_principal.pdf, Pág 3]."


def test_empty_response_is_returned_unchanged() -> None:
    rendered, stats = CitationRenderer([]).render("")

    assert rendered == ""
    assert stats.placeholders == 0
    assert stats.replaced == 0
