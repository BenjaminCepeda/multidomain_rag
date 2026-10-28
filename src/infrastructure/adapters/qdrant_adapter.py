# src/infrastructure/adapters/qdrant_adapter.py
import logging
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.domain.exceptions.core_errors import VectorStorageError
from src.domain.models.document import ManualChunk
from src.domain.models.embedding import EmbeddingVector
from src.domain.ports.embedding_service import IEmbeddingService
from src.domain.ports.vector_repository import IVectorRepository
from src.infrastructure.adapters.embedding_adapter import SentenceTransformerEmbeddingAdapter
from src.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)

MICRO_BATCH_SIZE = 250

QDRANT_DEFAULT_LOCAL_PATH = "./.qdrant_data"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


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
        self._local_fallback_path = (
            settings.qdrant_default_local_path.strip() or QDRANT_DEFAULT_LOCAL_PATH
        )
        try:
            from fastembed import SparseTextEmbedding

            self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        except Exception as exc:
            logger.warning("SparseEmbeddingUnavailable | falling back to dense-only: %s", exc)
            self._sparse_model = None

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
                self._client = QdrantClient(path=self._local_fallback_path)
        else:
            local_path = location or self._local_fallback_path
            logger.info(f"Usando Qdrant local en: {local_path}")
            self._client = QdrantClient(path=local_path)

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
                    vectors_config={
                        DENSE_VECTOR_NAME: VectorParams(
                            size=self._vector_dim, distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams())
                    },
                )
        except Exception as exc:
            raise VectorStorageError(
                "Imposible verificar o estructurar la coleccion en Qdrant.", exc
            ) from exc

    def ensure_collection(self, vector_dim: int) -> None:
        self._vector_dim = vector_dim
        self._ensure_collection()

    def upsert(self, chunks: Sequence[ManualChunk], vectors: Sequence[EmbeddingVector]) -> None:
        self._ensure_collection()
        if len(chunks) != len(vectors):
            raise VectorStorageError("chunks and vectors length mismatch.")

        total = len(chunks)
        for start in range(0, total, MICRO_BATCH_SIZE):
            end = min(start + MICRO_BATCH_SIZE, total)
            batch_chunks = list(chunks[start:end])
            batch_vectors = list(vectors[start:end])
            batch_texts = [chunk.content for chunk in batch_chunks]
            sparse_embeddings = []
            if self._sparse_model is not None:
                try:
                    sparse_embeddings = list(self._sparse_model.embed(batch_texts))
                except Exception:
                    sparse_embeddings = []
            points: list[PointStruct] = []
            for idx, (chunk, vec) in enumerate(zip(batch_chunks, batch_vectors, strict=False)):
                sparse_emb = sparse_embeddings[idx] if idx < len(sparse_embeddings) else None
                vector_payload: dict[str, object] = {DENSE_VECTOR_NAME: vec.to_list()}
                if sparse_emb is not None:
                    try:
                        sparse_dict = sparse_emb.as_dict()
                        indices = sparse_dict.get("indices") or sparse_dict.get("id") or []
                        values = sparse_dict.get("values") or sparse_dict.get("value") or []
                        if indices and values:
                            vector_payload[SPARSE_VECTOR_NAME] = SparseVector(
                                indices=[int(i) for i in indices],
                                values=[float(v) for v in values],
                            )
                    except Exception as exc:
                        logger.debug(
                            "SPARSE_VECTOR_SKIPPED | chunk_id=%s | error=%s", chunk.chunk_id, exc
                        )
                payload_dict = {
                    "chunk_id": str(chunk.chunk_id),
                    "content": chunk.content,
                    "is_visual": bool(chunk.is_visual),
                    "page": int(chunk.page),
                    "section": chunk.section,
                    "document_id": str(chunk.document_id) if chunk.document_id else None,
                    "source": chunk.metadata.get("source") or chunk.metadata.get("filename"),
                    "filename": chunk.metadata.get("filename")
                    or (
                        str(chunk.metadata.get("source")).split("/")[-1]
                        if chunk.metadata.get("source")
                        else None
                    ),
                }
                for k, v in chunk.metadata.items():
                    if isinstance(v, str | int | float | bool) and k not in payload_dict:
                        payload_dict[k] = v
                    elif k == "image_paths" and isinstance(v, str):
                        payload_dict["image_paths"] = v
                import uuid

                from src.domain.models.ids import ChunkId, DocumentId

                doc_id_for_hash = (
                    chunk.document_id
                    if chunk.document_id is not None
                    else DocumentId.from_source(str(uuid.uuid4()))
                )
                img_paths_hash = payload_dict.get("image_paths")
                img_h = img_paths_hash if isinstance(img_paths_hash, str) else None
                point_id = ChunkId.from_content(
                    doc_id_for_hash,
                    str(chunk.content),
                    int(chunk.page),
                    chunk.section,
                    img_h,
                )
                try:
                    point_id_int = abs(hash(str(point_id))) % (10**12)
                except Exception:
                    point_id_int = start + idx
                points.append(
                    PointStruct(
                        id=point_id_int,
                        vector=vector_payload,  # type: ignore[arg-type]
                        payload=payload_dict,
                    )
                )
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
        import time

        retrieval_start = time.perf_counter()
        self._ensure_collection()
        try:
            dense_results = self._client.search(
                collection_name=self._collection_name,
                query_vector=(DENSE_VECTOR_NAME, query_vector.to_list()),
                limit=limit * 4,
            )
        except Exception:
            try:
                dense_results = self._client.search(
                    collection_name=self._collection_name,
                    query_vector=query_vector.to_list(),
                    limit=limit * 4,
                )
            except Exception as exc:
                logger.error(f"Error critico en ejecucion vectorial: {exc!s}")
                raise VectorStorageError(
                    "Fallo en la ejecucion de consulta vectorial.", exc
                ) from exc

        sparse_results = []
        if self._sparse_model is not None:
            try:
                from qdrant_client.models import NamedSparseVector, SparseVector

                sparse_embeddings = list(self._sparse_model.embed([""]))
                sparse_dict = sparse_embeddings[0].as_dict()
                indices = sparse_dict.get("indices") or sparse_dict.get("id") or []
                values = sparse_dict.get("values") or sparse_dict.get("value") or []
                if indices and values:
                    sparse_results = self._client.search(
                        collection_name=self._collection_name,
                        query_vector=NamedSparseVector(
                            name=SPARSE_VECTOR_NAME,
                            vector=SparseVector(
                                indices=[int(i) for i in indices],
                                values=[float(v) for v in values],
                            ),
                        ),
                        limit=limit * 4,
                    )
            except Exception as exc:
                logger.warning(f"Sparse search failed, falling back to dense only: {exc}")

        scores: dict[str, float] = {}
        payloads: dict[str, dict[str, object]] = {}
        for rank, hit in enumerate(dense_results):
            key = str(hit.id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (60.0 + rank + 1.0)
            payloads[key] = hit.payload or {}
        for rank, hit in enumerate(sparse_results):
            key = str(hit.id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (60.0 + rank + 1.0)
            if key not in payloads:
                payloads[key] = hit.payload or {}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        retrieval_elapsed = time.perf_counter() - retrieval_start
        logger.info("HYBRID_SEARCH_TIMING | retrieval_s=%.3f", retrieval_elapsed)
        valid_results: list[ManualChunk] = []
        from src.domain.models.ids import ChunkId

        for key, _rrf_score in ranked:
            payload_raw = payloads.get(key) or {}
            chunk_id_raw = payload_raw.get("chunk_id") or key
            page_val = payload_raw.get("page", 1)
            try:
                if page_val is None:
                    page_int = 1
                elif isinstance(page_val, int):
                    page_int = page_val
                elif isinstance(page_val, float):
                    page_int = int(page_val)
                else:
                    page_int = int(str(page_val))
            except Exception:
                page_int = 1
            section_val = payload_raw.get("section")
            section_str = str(section_val) if section_val is not None else None
            metadata_dict: dict[str, str | int | float | bool] = {}
            for k, v in payload_raw.items():
                if isinstance(v, str | int | float | bool):
                    metadata_dict[k] = v
            valid_results.append(
                ManualChunk(
                    chunk_id=ChunkId(str(chunk_id_raw)),
                    content=str(payload_raw.get("content", "")),
                    is_visual=bool(payload_raw.get("is_visual", False)),
                    page=page_int,
                    section=section_str,
                    metadata=metadata_dict,
                )
            )
        return valid_results

    def count(self) -> int:
        self._ensure_collection()
        return self._client.count(collection_name=self._collection_name).count

    def exists_by_id(self, chunk_id: str) -> bool:
        return False

    def clear(self) -> None:
        try:
            if self._client.collection_exists(collection_name=self._collection_name):
                self._client.delete_collection(collection_name=self._collection_name)
                logger.info("QDRANT_PURGE | collection=%s | action=deleted", self._collection_name)
        except Exception as exc:
            logger.error(
                "QDRANT_PURGE | collection=%s | action=delete_failed | error=%s",
                self._collection_name,
                exc,
            )
            raise VectorStorageError("Fallo al limpiar la coleccion vectorial.", exc) from exc
        self._ensure_collection()
        logger.info("QDRANT_PURGE | collection=%s | action=recreated_empty", self._collection_name)

    def list_sources(self) -> list[str]:
        self._ensure_collection()
        sources: set[str] = set()
        offset = None
        seen_offsets: set[object] = set()
        try:
            while True:
                records, next_offset = self._client.scroll(
                    collection_name=self._collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for record in records:
                    payload = record.payload or {}
                    source = payload.get("source")
                    if source:
                        sources.add(str(source))
                if next_offset is None or next_offset == offset or next_offset in seen_offsets:
                    break
                seen_offsets.add(next_offset)
                offset = next_offset
        except Exception as exc:
            raise VectorStorageError("Fallo al listar fuentes vectoriales.", exc) from exc
        return sorted(sources)

    def delete_by_source(self, source: str) -> None:
        self._ensure_collection()
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        try:
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="source", match=MatchValue(value=source)),
                    ]
                ),
            )
        except Exception as exc:
            raise VectorStorageError("Fallo al eliminar la fuente vectorial.", exc) from exc

    # --- Backward-compatible helpers ---
    def save_chunks(self, chunks: list[ManualChunk]) -> None:
        vectors = self._embedding.embed_batch([c.content for c in chunks])
        self.upsert(chunks, vectors)

    def search_similar(self, query: str, limit: int = 3) -> list[ManualChunk]:
        query_vector = self._embedding.embed_text(query)
        return self.search(query_vector, limit=limit, min_score=0.20)
