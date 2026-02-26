"""RAGEngine - orquestador principal del sistema RAG."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from rag_engine.chain_factory import build_rag_chain
from rag_engine.config import ConfigRAGElements
from rag_engine.llm_factory import create_embeddings, create_llm
from rag_engine.models import SearchResponse, SourceDocument
from rag_engine.retriever_factory import build_retriever
from rag_engine.vectordb import create_vectorstore_provider

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.runnables import Runnable
    from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class RAGEngine:
    """Orquestador principal del sistema RAG.

    Inmutable una vez creado: para cambiar configuracion, crear una nueva instancia.
    Esta es una decision consciente para v1 — simplifica el estado interno y evita
    bugs de reconfiguracion parcial.

    Se pueden inyectar objetos LLM/embeddings/vectorstore pre-configurados
    para testing o cuando se necesitan API keys explicitas.
    """

    def __init__(
        self,
        config: ConfigRAGElements | None = None,
        *,
        embeddings: Embeddings | None = None,
        llm_generation: BaseChatModel | None = None,
        llm_queries: BaseChatModel | None = None,
        vectorstore: VectorStore | None = None,
    ) -> None:
        self._config = config or ConfigRAGElements()

        # Embeddings (inyectados o creados desde config)
        self._embeddings = embeddings or create_embeddings(self._config.llm)

        # LLMs
        self._llm_generation = llm_generation or create_llm(self._config.llm)
        self._llm_queries = llm_queries or create_llm(
            self._config.llm,
            model=self._config.llm.query_model,
            temperature=self._config.llm.query_temperature,
        )

        # VectorStore
        if vectorstore is not None:
            self._vectorstore = vectorstore
        else:
            provider = create_vectorstore_provider(self._config.vector_db.db_type)
            self._vectorstore = provider.get_vectorstore(self._config.vector_db, self._embeddings)

        # Retriever
        self._retriever: BaseRetriever = build_retriever(
            self._vectorstore,
            self._config.retriever,
            self._llm_queries,
            multi_query_prompt=self._config.multi_query_prompt,
        )

        # Cadena RAG (single-pass)
        self._chain: Runnable = build_rag_chain(
            self._retriever,
            self._llm_generation,
            rag_template=self._config.rag_template,
        )

        logger.info("RAGEngine inicializado correctamente")

    # -- Propiedades de solo lectura ------------------------------------------

    @property
    def config(self) -> ConfigRAGElements:
        return self._config

    @property
    def vectorstore(self) -> VectorStore:
        return self._vectorstore

    @property
    def retriever(self) -> BaseRetriever:
        return self._retriever

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    # -- API publica ----------------------------------------------------------

    def query(self, question: str) -> SearchResponse:
        """Ejecuta una consulta RAG completa y retorna un SearchResponse."""
        try:
            result: dict[str, Any] = self._chain.invoke({"question": question})

            source_docs = [
                SourceDocument(
                    content=doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                    source=os.path.basename(doc.metadata.get("source", "desconocida")),
                    page=doc.metadata.get("page", "N/A"),
                    metadata=doc.metadata,
                )
                for doc in result.get("source_documents", [])
            ]

            return SearchResponse(
                answer=result["answer"],
                source_documents=source_docs,
            )

        except Exception as e:
            logger.exception("Error al procesar consulta: %s", question)
            return SearchResponse(answer="", error=str(e))

    def get_retriever_info(self) -> dict[str, Any]:
        """Retorna informacion sobre la configuracion actual del retriever."""
        rc = self._config.retriever
        search_label = rc.search_type.value.upper() + " + MultiQuery"
        if rc.enable_hybrid_search:
            search_label += " + Hybrid"

        return {
            "tipo": search_label,
            "documentos": rc.search_k,
            "diversidad": rc.mmr_diversity_lambda,
            "candidatos": rc.mmr_fetch_k,
            "umbral": rc.similarity_threshold if rc.enable_hybrid_search else "N/A",
        }
