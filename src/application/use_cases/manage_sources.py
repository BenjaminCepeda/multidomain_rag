# src/application/use_cases/manage_sources.py
from src.domain.ports.vector_repository import IVectorRepository


class ManageSourcesUseCase:
    """Use case for listing and deleting ingested sources from vector storage."""

    def __init__(self, repository: IVectorRepository) -> None:
        self._repository = repository

    def list_sources(self) -> list[str]:
        return self._repository.list_sources()

    def delete_source(self, source: str) -> None:
        self._repository.delete_by_source(source)

    def clear_all(self) -> None:
        self._repository.clear()
