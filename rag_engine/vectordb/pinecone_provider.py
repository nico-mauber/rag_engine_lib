"""Proveedor de Pinecone."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_engine.vectordb.base import VectorStoreProvider

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore

    from rag_engine.config import VectorDBConfig

logger = logging.getLogger(__name__)


class PineconeProvider(VectorStoreProvider):
    """Proveedor que conecta con un indice de Pinecone.

    Requiere PINECONE_API_KEY en variables de entorno.
    """

    def get_vectorstore(
        self,
        config: VectorDBConfig,
        embeddings: Embeddings,
    ) -> VectorStore:
        from langchain_pinecone import PineconeVectorStore

        logger.info("Conectando a Pinecone index=%s", config.pinecone_index_name)
        kwargs: dict = {
            "index_name": config.pinecone_index_name,
            "embedding": embeddings,
        }
        if config.pinecone_api_key:
            kwargs["pinecone_api_key"] = config.pinecone_api_key
        return PineconeVectorStore(**kwargs)
