"""Configuracion del engine RAG mediante dataclasses compuestas."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    GOOGLE = "google"
    OPENAI = "openai"


class VectorDBType(str, Enum):
    CHROMA = "chroma"
    PINECONE = "pinecone"


class SearchType(str, Enum):
    MMR = "mmr"
    SIMILARITY = "similarity"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    """Configuracion de modelos LLM y embeddings.

    api_key: Si se proporciona, se usa directamente. Si es None,
    el proveedor la busca en variables de entorno (GOOGLE_API_KEY, OPENAI_API_KEY).
    """

    provider: LLMProvider = LLMProvider.GOOGLE
    embedding_model: str = "models/gemini-embedding-001"
    query_model: str = "models/gemini-2.5-flash"
    generation_model: str = "models/gemini-2.5-flash"
    query_temperature: float = 0.0
    generation_temperature: float = 0.0
    api_key: Optional[str] = None


@dataclass(frozen=True)
class VectorDBConfig:
    """Configuracion del vector store."""

    db_type: VectorDBType = VectorDBType.CHROMA
    path: Optional[str] = None
    collection: str = "langchain"
    pinecone_index_name: Optional[str] = None
    pinecone_api_key: Optional[str] = None

    def __post_init__(self) -> None:
        if self.db_type == VectorDBType.CHROMA and not self.path:
            raise ValueError("VectorDBConfig.path es requerido cuando db_type es CHROMA")
        if self.db_type == VectorDBType.PINECONE and not self.pinecone_index_name:
            raise ValueError(
                "VectorDBConfig.pinecone_index_name es requerido cuando db_type es PINECONE"
            )


@dataclass(frozen=True)
class RetrieverConfig:
    """Configuracion del retriever."""

    search_type: SearchType = SearchType.MMR
    search_k: int = 2
    mmr_diversity_lambda: float = 0.7
    mmr_fetch_k: int = 20
    enable_hybrid_search: bool = True
    similarity_threshold: float = 0.70
    ensemble_weights: tuple[float, float] = (0.7, 0.3)


@dataclass(frozen=True)
class SplitterConfig:
    """Configuracion del text splitter para ingesta."""

    chunk_size: int = 5000
    chunk_overlap: int = 1000


# ---------------------------------------------------------------------------
# Config principal (compuesta)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigRAGElements:
    """Configuracion completa del engine RAG.

    Se compone de sub-dataclasses para evitar un god object.
    Las API keys se resuelven desde variables de entorno por cada proveedor
    (GOOGLE_API_KEY, OPENAI_API_KEY, etc.), siguiendo el patron estandar de LangChain.
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = field(default_factory=lambda: VectorDBConfig(path="./chroma_db"))
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
    rag_template: Optional[str] = None
    multi_query_prompt: Optional[str] = None
