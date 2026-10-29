from pathlib import Path

import fitz

from src.infrastructure.adapters.pymupdf_adapter import PyMuPDFAdapter
from src.infrastructure.config.settings import Settings

REQUIRED_VISUAL_KEYS = {
    "document_name",
    "document_title",
    "document_context_summary",
    "page_number",
    "image_title",
    "theme_summary",
    "surrounding_text_context",
    "image_path",
}


def _make_pdf(path: Path, title: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Inferencia en el borde", fontsize=16)
    page.insert_text(
        (72, 100),
        "Este capitulo describe como se realiza la inferencia en el borde con modelos TinyML.",
        fontsize=11,
    )
    page.insert_text((72, 150), "Figura 1: Flujo de inferencia en el borde", fontsize=10)

    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 24, 24), False)
    pixmap.set_rect(pixmap.irect, (200, 30, 30))
    page.insert_image(fitz.Rect(72, 170, 180, 260), pixmap=pixmap)

    doc.set_metadata({"title": title})
    doc.save(str(path))
    doc.close()


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_files_path=str(tmp_path / "data"),
        raw_files_path="raw",
        extracted_images_path="extracted_images",
    )


def test_visual_chunks_carry_macro_and_micro_context(tmp_path: Path) -> None:
    pdf_path = tmp_path / "TinyML_en_el_borde.pdf"
    _make_pdf(pdf_path, "TinyML en el borde")
    settings = _settings(tmp_path)

    chunks = PyMuPDFAdapter(settings).extract_chunks(str(pdf_path))
    visuals = [chunk for chunk in chunks if chunk.is_visual]

    assert len(visuals) == 1
    metadata = visuals[0].metadata
    assert REQUIRED_VISUAL_KEYS.issubset(metadata)
    assert metadata["document_name"] == "TinyML_en_el_borde.pdf"
    assert metadata["document_title"] == "TinyML en el borde"
    assert metadata["page_number"] == 1
    assert "inferencia" in str(metadata["image_title"]).lower()
    assert "inferencia" in str(metadata["theme_summary"]).lower()
    assert "inferencia" in str(metadata["surrounding_text_context"]).lower()
    assert "TinyML en el borde" in visuals[0].content

    image_path = Path(str(metadata["image_path"]))
    assert image_path.is_file()
    assert image_path.is_relative_to(settings.extracted_images_dir.resolve())


def test_text_chunks_reference_document_context(tmp_path: Path) -> None:
    pdf_path = tmp_path / "EdgeAI_manual.pdf"
    _make_pdf(pdf_path, "")
    settings = _settings(tmp_path)

    chunks = PyMuPDFAdapter(settings).extract_chunks(str(pdf_path))
    text_chunks = [chunk for chunk in chunks if not chunk.is_visual]

    assert text_chunks
    assert all(chunk.metadata["document_title"] == "EdgeAI manual" for chunk in text_chunks)
    assert all(chunk.metadata["page_number"] == 1 for chunk in text_chunks)


def test_visual_chunk_ids_do_not_collide_with_text_chunks(tmp_path: Path) -> None:
    pdf_path = tmp_path / "EdgeAI_manual.pdf"
    _make_pdf(pdf_path, "EdgeAI manual")
    settings = _settings(tmp_path)

    chunks = PyMuPDFAdapter(settings).extract_chunks(str(pdf_path))
    chunk_ids = [str(chunk.chunk_id) for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))
