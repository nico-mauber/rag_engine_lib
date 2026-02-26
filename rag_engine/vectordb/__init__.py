"""Factory para crear proveedores de vector store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_engine.config import VectorDBType

if TYPE_CHECKING:
    from rag_engine.vectordb.base import VectorStoreProvider


def create_vectorstore_provider(db_type: VectorDBType) -> VectorStoreProvider:
    """Retorna el proveedor correspondiente al tipo de vector DB."""
    if db_type == VectorDBType.CHROMA:
        from rag_engine.vectordb.chroma_provider import ChromaProvider

        return ChromaProvider()

    if db_type == VectorDBType.PINECONE:
        from rag_engine.vectordb.pinecone_provider import PineconeProvider

        return PineconeProvider()

    raise ValueError(f"Tipo de vector DB no soportado: {db_type}")
