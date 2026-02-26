"""DocumentIngester — ingesta de documentos PDF a un vector store."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_engine.config import ConfigRAGElements, VectorDBType
from rag_engine.llm_factory import create_embeddings
from rag_engine.vectordb import create_vectorstore_provider

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class DocumentIngester:
    """Ingesta documentos PDF en un vector store.

    Puede usarse de forma standalone (sin RAGEngine) o recibiendo
    objetos pre-configurados.

    Ejemplo::

        ingester = DocumentIngester(config=config)
        ingester.ingest_pdf_directory("./contratos")
    """

    def __init__(
        self,
        config: ConfigRAGElements | None = None,
        *,
        embeddings: Embeddings | None = None,
        vectorstore: VectorStore | None = None,
    ) -> None:
        self._config = config or ConfigRAGElements()
        self._embeddings = embeddings or create_embeddings(self._config.llm)

        if vectorstore is not None:
            self._vectorstore = vectorstore
        else:
            provider = create_vectorstore_provider(self._config.vector_db.db_type)
            self._vectorstore = provider.get_vectorstore(self._config.vector_db, self._embeddings)

    @property
    def vectorstore(self) -> VectorStore:
        return self._vectorstore

    def ingest_pdf_directory(self, directory: str) -> int:
        """Carga todos los PDFs de un directorio, los divide y agrega al vector store.

        Returns:
            Cantidad de chunks ingresados.
        """
        from langchain_community.document_loaders import PyPDFDirectoryLoader

        loader = PyPDFDirectoryLoader(directory)
        documents = loader.load()
        logger.info("Se cargaron %d paginas desde %s", len(documents), directory)

        return self.ingest_documents(documents)

    def ingest_documents(self, documents: list[Document]) -> int:
        """Divide documentos en chunks y los agrega al vector store.

        Returns:
            Cantidad de chunks ingresados.
        """
        sc = self._config.splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=sc.chunk_size,
            chunk_overlap=sc.chunk_overlap,
        )

        chunks = splitter.split_documents(documents)
        logger.info("Se crearon %d chunks (size=%d, overlap=%d)", len(chunks), sc.chunk_size, sc.chunk_overlap)

        self._vectorstore.add_documents(chunks)
        logger.info("Chunks ingresados al vector store")

        return len(chunks)
