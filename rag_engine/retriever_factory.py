"""Factory para construir retrievers con distintas estrategias de busqueda."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.prompts import PromptTemplate

from rag_engine.config import SearchType
from rag_engine.prompts import MULTI_QUERY_PROMPT

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.vectorstores import VectorStore

    from rag_engine.config import RetrieverConfig

logger = logging.getLogger(__name__)


def build_retriever(
    vectorstore: VectorStore,
    config: RetrieverConfig,
    llm_queries: BaseChatModel,
    *,
    multi_query_prompt: str | None = None,
) -> BaseRetriever:
    """Construye el retriever segun la configuracion.

    Soporta MMR, similarity, MultiQuery y Ensemble (hibrido).
    """
    search_kwargs: dict = {"k": config.search_k}

    if config.search_type == SearchType.MMR:
        search_kwargs["lambda_mult"] = config.mmr_diversity_lambda
        search_kwargs["fetch_k"] = config.mmr_fetch_k

    base_retriever = vectorstore.as_retriever(
        search_type=config.search_type.value,
        search_kwargs=search_kwargs,
    )
    logger.info("Base retriever: %s (k=%d)", config.search_type.value, config.search_k)

    # MultiQueryRetriever sobre el base retriever
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever

    prompt_text = multi_query_prompt or MULTI_QUERY_PROMPT
    mq_prompt = PromptTemplate.from_template(prompt_text)

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_queries,
        prompt=mq_prompt,
    )
    logger.info("MultiQueryRetriever configurado")

    if not config.enable_hybrid_search:
        return multi_query_retriever

    # Ensemble (hibrido): combina MultiQuery + similarity
    from langchain_classic.retrievers import EnsembleRetriever

    similarity_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.search_k},
    )

    w1, w2 = config.ensemble_weights
    ensemble = EnsembleRetriever(
        retrievers=[multi_query_retriever, similarity_retriever],
        weights=[w1, w2],
        similarity_threshold=config.similarity_threshold,
    )
    logger.info("EnsembleRetriever hibrido: pesos=(%.1f, %.1f), umbral=%.2f", w1, w2, config.similarity_threshold)
    return ensemble
