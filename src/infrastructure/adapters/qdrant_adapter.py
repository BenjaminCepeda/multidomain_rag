# src/infrastructure/adapters/qdrant_adapter.py
import os
import logging
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from src.domain.ports.vector_repository import IVectorRepository
from src.domain.models.document import ManualChunk
from src.domain.exceptions.core_errors import VectorStorageError

logger = logging.getLogger(__name__)

class QdrantAdapter(IVectorRepository):
    """Adaptador profesional con verificación de integridad, resiliencia y auditoría de búsqueda."""

    def __init__(self, location: str, collection_name: str, model_name: str, vector_dim: int) -> None:
        self._collection_name = collection_name
        self._vector_dim = vector_dim
        
        # Conexión con resiliencia: Intento remoto -> Fallback local
        try:
            self._client = QdrantClient(url=location, timeout=2.0)
            self._client.get_collections()
            logger.info(f"Conectado a Qdrant remoto en: {location}")
        except Exception as e:
            logger.warning(f"No se pudo conectar a Qdrant remoto: {e}. Cambiando a almacenamiento local.")
            self._client = QdrantClient(path="./qdrant_local_data")
            
        cache_folder = os.getenv("HF_MODEL_CACHE")
        self._encoder = SentenceTransformer(model_name, cache_folder=cache_folder)

    def has_data(self) -> bool:
        """Verifica existencia de datos para habilitar/bloquear la UI."""
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
            raise VectorStorageError("Imposible verificar o estructurar la colección en Qdrant.", exc)

    def save_chunks(self, chunks: List[ManualChunk]) -> None:
        self._ensure_collection()
        points = [
            PointStruct(
                id=idx,
                vector=self._encoder.encode(chunk.content).tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "is_visual": chunk.is_visual,
                    **chunk.metadata
                }
            ) for idx, chunk in enumerate(chunks)
        ]
        try:
            self._client.upsert(collection_name=self._collection_name, points=points)
            logger.info(f"Indexación exitosa: {len(points)} fragmentos guardados en '{self._collection_name}'.")
        except Exception as exc:
            logger.error(f"Error crítico en persistencia: {exc}")
            raise VectorStorageError("Fallo en la persistencia vectorial.", exc)

    def search_similar(self, query: str, limit: int = 3) -> List[ManualChunk]:
        """Realiza búsqueda con auditoría de logs para diagnóstico de alucinaciones/vacío."""
        self._ensure_collection()
        try:
            query_vector = self._encoder.encode(query).tolist()
            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            # Auditoría detallada de cada intento de búsqueda
            if not results:
                logger.warning(f"AUDITORÍA: Búsqueda vacía para la consulta: '{query}'")
            else:
                logger.info(f"AUDITORÍA: Se encontraron {len(results)} candidatos para '{query}'")
                for i, hit in enumerate(results):
                    logger.info(f" -> [{i}] Score: {hit.score:.4f} | Origen: {hit.payload.get('source', 'N/A')} Pág: {hit.payload.get('page', 'N/A')} | Texto: {str(hit.payload.get('content'))[:70]}...")
            
            # Filtro de seguridad: retornamos solo los que tienen relevancia técnica (score >= 0.20)
            valid_results = [
                ManualChunk(
                    chunk_id=str(hit.payload.get("chunk_id")),
                    content=str(hit.payload.get("content")),
                    is_visual=bool(hit.payload.get("is_visual")),
                    metadata=hit.payload
                ) for hit in results if hit.score >= 0.20 
            ]
            
            if not valid_results and results:
                logger.warning(f"AUDITORÍA: Coincidencias encontradas pero descartadas por bajo score (umbral 0.20).")
                
            return valid_results

        except Exception as exc:
            logger.error(f"Error crítico en ejecución vectorial: {str(exc)}")
            raise VectorStorageError("Fallo en la ejecución de consulta vectorial.", exc)