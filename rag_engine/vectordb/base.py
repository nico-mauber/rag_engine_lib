"""ABC para proveedores de vector store."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore

    from rag_engine.config import VectorDBConfig


class VectorStoreProvider(ABC):
    """Interfaz abstracta que cada proveedor de vector store debe implementar."""

    @abstractmethod
    def get_vectorstore(
        self,
        config: VectorDBConfig,
        embeddings: Embeddings,
    ) -> VectorStore:
        """Retorna una instancia de VectorStore lista para consultas."""
