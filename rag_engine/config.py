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

    Campos obligatorios (el usuario debe proporcionarlos siempre):
        provider, embedding_model, query_model, generation_model, api_key

    Campos opcionales (tienen valores por defecto):
        query_temperature, generation_temperature
    """

    # --- Obligatorios (sin default → error si no se pasan) ---
    provider: LLMProvider
    embedding_model: str
    query_model: str
    generation_model: str
    api_key: str

    # --- Opcionales (con default) ---
    query_temperature: float = 0.0
    generation_temperature: float = 0.0


@dataclass(frozen=True)
class VectorDBConfig:
    """Configuracion del vector store.

    Campos obligatorios (el usuario debe proporcionarlos siempre):
        db_type, collection

    Campos condicionales (validados en __post_init__):
        path → requerido si db_type es CHROMA
        pinecone_index_name → requerido si db_type es PINECONE

    Campos opcionales:
        pinecone_api_key
    """

    # --- Obligatorios ---
    db_type: VectorDBType
    collection: str

    # --- Condicionales (dependen del db_type) ---
    path: Optional[str] = None
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
    """Configuracion del retriever. Todos los campos son opcionales."""

    search_type: SearchType = SearchType.MMR
    search_k: int = 2
    mmr_diversity_lambda: float = 0.7
    mmr_fetch_k: int = 20
    enable_hybrid_search: bool = True
    similarity_threshold: float = 0.70
    ensemble_weights: tuple[float, float] = (0.7, 0.3)


@dataclass(frozen=True)
class SplitterConfig:
    """Configuracion del text splitter para ingesta. Todos los campos son opcionales."""

    chunk_size: int = 5000
    chunk_overlap: int = 1000


# ---------------------------------------------------------------------------
# Config principal (compuesta)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigRAGElements:
    """Configuracion completa del engine RAG.

    Campos obligatorios:
        llm → LLMConfig (el usuario debe crearlo explicitamente)
        vector_db → VectorDBConfig (el usuario debe crearlo explicitamente)

    Campos opcionales:
        retriever, splitter, rag_template, multi_query_prompt
    """

    # --- Obligatorios (sin default → error si no se pasan) ---
    llm: LLMConfig
    vector_db: VectorDBConfig

    # --- Opcionales (con default) ---
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
    rag_template: Optional[str] = None
    multi_query_prompt: Optional[str] = None
