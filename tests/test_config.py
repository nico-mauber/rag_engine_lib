"""Tests para el modulo de configuracion."""

import pytest

from rag_engine.config import (
    ConfigRAGElements,
    LLMConfig,
    LLMProvider,
    RetrieverConfig,
    SearchType,
    SplitterConfig,
    VectorDBConfig,
    VectorDBType,
)


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.provider == LLMProvider.GOOGLE
        assert cfg.embedding_model == "models/gemini-embedding-001"
        assert cfg.query_temperature == 0.0

    def test_custom_values(self):
        cfg = LLMConfig(provider=LLMProvider.OPENAI, embedding_model="text-embedding-3-small")
        assert cfg.provider == LLMProvider.OPENAI
        assert cfg.embedding_model == "text-embedding-3-small"

    def test_frozen(self):
        cfg = LLMConfig()
        with pytest.raises(AttributeError):
            cfg.provider = LLMProvider.OPENAI  # type: ignore[misc]


class TestVectorDBConfig:
    def test_chroma_requires_path(self):
        with pytest.raises(ValueError, match="path es requerido"):
            VectorDBConfig(db_type=VectorDBType.CHROMA, path=None)

    def test_chroma_with_path(self):
        cfg = VectorDBConfig(db_type=VectorDBType.CHROMA, path="/tmp/chroma")
        assert cfg.path == "/tmp/chroma"

    def test_pinecone_requires_index_name(self):
        with pytest.raises(ValueError, match="pinecone_index_name es requerido"):
            VectorDBConfig(db_type=VectorDBType.PINECONE)

    def test_pinecone_with_index(self):
        cfg = VectorDBConfig(
            db_type=VectorDBType.PINECONE,
            pinecone_index_name="my-index",
        )
        assert cfg.pinecone_index_name == "my-index"


class TestRetrieverConfig:
    def test_defaults(self):
        cfg = RetrieverConfig()
        assert cfg.search_type == SearchType.MMR
        assert cfg.search_k == 2
        assert cfg.enable_hybrid_search is True
        assert cfg.ensemble_weights == (0.7, 0.3)


class TestSplitterConfig:
    def test_defaults(self):
        cfg = SplitterConfig()
        assert cfg.chunk_size == 5000
        assert cfg.chunk_overlap == 1000


class TestConfigRAGElements:
    def test_defaults(self):
        cfg = ConfigRAGElements()
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.vector_db, VectorDBConfig)
        assert isinstance(cfg.retriever, RetrieverConfig)
        assert isinstance(cfg.splitter, SplitterConfig)
        assert cfg.rag_template is None
        assert cfg.multi_query_prompt is None

    def test_custom_sub_configs(self):
        cfg = ConfigRAGElements(
            llm=LLMConfig(provider=LLMProvider.OPENAI),
            vector_db=VectorDBConfig(path="/tmp/test"),
            retriever=RetrieverConfig(search_k=5),
        )
        assert cfg.llm.provider == LLMProvider.OPENAI
        assert cfg.retriever.search_k == 5

    def test_frozen(self):
        cfg = ConfigRAGElements()
        with pytest.raises(AttributeError):
            cfg.rag_template = "test"  # type: ignore[misc]
