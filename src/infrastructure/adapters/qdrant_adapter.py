# src/infrastructure/adapters/qdrant_adapter.py
import logging
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.domain.exceptions.core_errors import VectorStorageError
from src.domain.models.document import ManualChunk
from src.domain.models.embedding import EmbeddingVector
from src.domain.ports.embedding_service import IEmbeddingService
from src.domain.ports.vector_repository import IVectorRepository
from src.infrastructure.adapters.embedding_adapter import SentenceTransformerEmbeddingAdapter
from src.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)

MICRO_BATCH_SIZE = 250


class QdrantAdapter(IVectorRepository):
    """Qdrant adapter with centralized settings and micro-batched upserts."""

    def __init__(
        self,
        settings: Settings,
        embedding: IEmbeddingService | None = None,
    ) -> None:
        self._settings = settings
        self._collection_name = settings.qdrant_collection_name
        self._vector_dim = settings.embedding_vector_dim
        self._embedding: IEmbeddingService = embedding or SentenceTransformerEmbeddingAdapter(
            settings=settings
        )

        location = settings.qdrant_url.strip()
        if location.startswith("http://") or location.startswith("https://"):
            try:
                self._client = QdrantClient(
                    url=location, timeout=int(settings.qdrant_timeout_seconds)
                )
                self._client.get_collections()
                logger.info(f"Conectado a Qdrant remoto en: {location}")
            except Exception as e:
                logger.warning(
                    f"No se pudo conectar a Qdrant remoto: {e}. Cambiando a almacenamiento local."
                )
                self._client = QdrantClient(path="./qdrant_local_data")
        else:
            logger.info(f"Usando Qdrant local en: {location}")
            self._client = QdrantClient(path=location)

    def has_data(self) -> bool:
        try:
            if not self._client.collection_exists(collection_name=self._collection_name):
                return False
            count = self._client.count(collection_name=self._collection_name)
            return count.count > 0
        except Exception:
            return False

    def _ensure_collection(self) -> None:
        try:
            if not self._client.collection_exists(collection_name=self._collection_name):
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=self._vector_dim, distance=Distance.COSINE),
                )
        except Exception as exc:
            raise VectorStorageError(
                "Imposible verificar o estructurar la coleccion en Qdrant.", exc
            ) from exc

    def ensure_collection(self, vector_dim: int) -> None:
        self._vector_dim = vector_dim
        self._ensure_collection()

    def upsert(
        self, chunks: Sequence[ManualChunk], vectors: Sequence[EmbeddingVector]
    ) -> None:
        self._ensure_collection()
        if len(chunks) != len(vectors):
            raise VectorStorageError("chunks and vectors length mismatch.")

        total = len(chunks)
        for start in range(0, total, MICRO_BATCH_SIZE):
            end = min(start + MICRO_BATCH_SIZE, total)
            batch_chunks = list(chunks[start:end])
            batch_vectors = list(vectors[start:end])
            points = [
                PointStruct(
                    id=start + idx,
                    vector=vec.to_list(),
                    payload={
                        "chunk_id": str(chunk.chunk_id),
                        "content": chunk.content,
                        "is_visual": chunk.is_visual,
                        "page": chunk.page,
                        "section": chunk.section,
                        "document_id": str(chunk.document_id) if chunk.document_id else None,
                        **chunk.metadata,
                    },
                )
                for idx, (chunk, vec) in enumerate(zip(batch_chunks, batch_vectors, strict=False))
            ]
            try:
                self._client.upsert(collection_name=self._collection_name, points=points)
                logger.info(
                    f"Indexacion exitosa: {len(points)} fragmentos guardados en "
                    f"'{self._collection_name}' ({end}/{total})."
                )
            except Exception as exc:
                logger.error(f"Error critico en persistencia: {exc}")
                raise VectorStorageError("Fallo en la persistencia vectorial.", exc) from exc

    def search(
        self, query_vector: EmbeddingVector, limit: int, min_score: float
    ) -> list[ManualChunk]:
        self._ensure_collection()
        try:
            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vector.to_list(),
                limit=limit,
            )
            valid_results: list[ManualChunk] = []
            for hit in results:
                if hit.score < min_score:
                    continue
                payload = hit.payload or {}
                from src.domain.models.ids import ChunkId

                chunk_id_raw = payload.get("chunk_id") or str(hit.id)
                valid_results.append(
                    ManualChunk(
                        chunk_id=ChunkId(str(chunk_id_raw)),
                        content=str(payload.get("content", "")),
                        is_visual=bool(payload.get("is_visual", False)),
                        page=int(payload.get("page", 1)) if payload.get("page") is not None else 1,
                        section=payload.get("section"),
                        metadata=payload,
                    )
                )
            return valid_results
        except Exception as exc:
            logger.error(f"Error critico en ejecucion vectorial: {exc!s}")
            raise VectorStorageError("Fallo en la ejecucion de consulta vectorial.", exc) from exc

    def count(self) -> int:
        self._ensure_collection()
        return self._client.count(collection_name=self._collection_name).count

    def exists_by_id(self, chunk_id: str) -> bool:
        return False

    def clear(self) -> None:
        try:
            if self._client.collection_exists(collection_name=self._collection_name):
                self._client.delete_collection(collection_name=self._collection_name)
        except Exception as exc:
            raise VectorStorageError("Fallo al limpiar la coleccion vectorial.", exc) from exc
        self._ensure_collection()

    # --- Backward-compatible helpers ---
    def save_chunks(self, chunks: list[ManualChunk]) -> None:
        vectors = self._embedding.embed_batch([c.content for c in chunks])
        self.upsert(chunks, vectors)

    def search_similar(self, query: str, limit: int = 3) -> list[ManualChunk]:
        query_vector = self._embedding.embed_text(query)
        return self.search(query_vector, limit=limit, min_score=0.20)
