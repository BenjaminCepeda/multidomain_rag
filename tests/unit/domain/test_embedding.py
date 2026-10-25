import pytest

from src.domain.models.embedding import EmbeddingVector


def test_embedding_vector_valid() -> None:
    v = EmbeddingVector(values=(1.0, 2.0, 3.0))
    assert v.dimension == 3
    assert v.to_list() == [1.0, 2.0, 3.0]


def test_embedding_vector_empty() -> None:
    with pytest.raises(ValueError):
        EmbeddingVector(values=())


def test_embedding_vector_non_numeric() -> None:
    with pytest.raises(TypeError):
        EmbeddingVector(values=("a", "b"))  # type: ignore[arg-type]
