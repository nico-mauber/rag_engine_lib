"""Proveedor de ChromaDB."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_engine.vectordb.base import VectorStoreProvider

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore

    from rag_engine.config import VectorDBConfig

logger = logging.getLogger(__name__)


class ChromaProvider(VectorStoreProvider):
    """Proveedor que conecta con una instancia persistente de ChromaDB."""

    def get_vectorstore(
        self,
        config: VectorDBConfig,
        embeddings: Embeddings,
    ) -> VectorStore:
        from langchain_chroma import Chroma

        logger.info("Conectando a ChromaDB en %s (coleccion=%s)", config.path, config.collection)
        return Chroma(
            embedding_function=embeddings,
            persist_directory=config.path,
            collection_name=config.collection,
        )
