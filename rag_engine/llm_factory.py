"""Factories para crear instancias de LLM y embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from rag_engine.config import ConfigRAGElements, LLMProvider

if TYPE_CHECKING:
    from rag_engine.config import LLMConfig

logger = logging.getLogger(__name__)


def create_embeddings(config: LLMConfig) -> Embeddings:
    """Crea una instancia de embeddings segun el proveedor configurado.

    Las API keys se resuelven desde variables de entorno por LangChain.
    """
    if config.provider == LLMProvider.GOOGLE:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        logger.info("Creando embeddings Google: %s", config.embedding_model)
        return GoogleGenerativeAIEmbeddings(model=config.embedding_model)

    if config.provider == LLMProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings

        logger.info("Creando embeddings OpenAI: %s", config.embedding_model)
        return OpenAIEmbeddings(model=config.embedding_model)

    raise ValueError(f"Proveedor de embeddings no soportado: {config.provider}")


def create_llm(
    config: LLMConfig,
    *,
    model: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """Crea una instancia de LLM segun el proveedor configurado.

    Args:
        config: Configuracion LLM.
        model: Override del modelo (por defecto usa generation_model).
        temperature: Override de la temperatura (por defecto usa generation_temperature).
    """
    model = model or config.generation_model
    temperature = temperature if temperature is not None else config.generation_temperature

    if config.provider == LLMProvider.GOOGLE:
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info("Creando LLM Google: %s (temp=%.1f)", model, temperature)
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    if config.provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        logger.info("Creando LLM OpenAI: %s (temp=%.1f)", model, temperature)
        return ChatOpenAI(model=model, temperature=temperature)

    raise ValueError(f"Proveedor de LLM no soportado: {config.provider}")
