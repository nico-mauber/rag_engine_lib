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


# -- Helpers para crear configs validas rapidamente --

def _llm(**overrides):
    defaults = dict(
        provider=LLMProvider.GOOGLE,
        embedding_model="models/gemini-embedding-001",
        query_model="models/gemini-2.5-flash",
        generation_model="models/gemini-2.5-flash",
        api_key="test-key",
    )
    defaults.update(overrides)
    return LLMConfig(**defaults)


def _vdb(**overrides):
    defaults = dict(db_type=VectorDBType.CHROMA, collection="langchain", path="/tmp/chroma")
    defaults.update(overrides)
    return VectorDBConfig(**defaults)


class TestLLMConfig:
    def test_required_fields(self):
        cfg = _llm()
        assert cfg.provider == LLMProvider.GOOGLE
        assert cfg.api_key == "test-key"

    def test_missing_provider_raises(self):
        with pytest.raises(TypeError):
            LLMConfig(
                embedding_model="x", query_model="x",
                generation_model="x", api_key="x",
            )

    def test_missing_api_key_raises(self):
        with pytest.raises(TypeError):
            LLMConfig(
                provider=LLMProvider.GOOGLE,
                embedding_model="x", query_model="x",
                generation_model="x",
            )

    def test_optional_defaults(self):
        cfg = _llm()
        assert cfg.query_temperature == 0.0
        assert cfg.generation_temperature == 0.0

    def test_frozen(self):
        cfg = _llm()
        with pytest.raises(AttributeError):
            cfg.provider = LLMProvider.OPENAI  # type: ignore[misc]


class TestVectorDBConfig:
    def test_chroma_requires_path(self):
        with pytest.raises(ValueError, match="path es requerido"):
            VectorDBConfig(db_type=VectorDBType.CHROMA, collection="test", path=None)

    def test_chroma_ok(self):
        cfg = VectorDBConfig(db_type=VectorDBType.CHROMA, collection="docs", path="/tmp/chroma")
        assert cfg.path == "/tmp/chroma"

    def test_pinecone_requires_index_name(self):
        with pytest.raises(ValueError, match="pinecone_index_name es requerido"):
            VectorDBConfig(db_type=VectorDBType.PINECONE, collection="test")

    def test_pinecone_ok(self):
        cfg = VectorDBConfig(
            db_type=VectorDBType.PINECONE,
            collection="test",
            pinecone_index_name="my-index",
            pinecone_api_key="pk-123",
        )
        assert cfg.pinecone_index_name == "my-index"

    def test_missing_db_type_raises(self):
        with pytest.raises(TypeError):
            VectorDBConfig(collection="test", path="/tmp")


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
    def test_requires_llm_and_vector_db(self):
        with pytest.raises(TypeError):
            ConfigRAGElements()

    def test_ok_with_required_fields(self):
        cfg = ConfigRAGElements(llm=_llm(), vector_db=_vdb())
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.vector_db, VectorDBConfig)
        assert isinstance(cfg.retriever, RetrieverConfig)
        assert isinstance(cfg.splitter, SplitterConfig)

    def test_optional_defaults(self):
        cfg = ConfigRAGElements(llm=_llm(), vector_db=_vdb())
        assert cfg.rag_template is None
        assert cfg.multi_query_prompt is None

    def test_frozen(self):
        cfg = ConfigRAGElements(llm=_llm(), vector_db=_vdb())
        with pytest.raises(AttributeError):
            cfg.rag_template = "test"  # type: ignore[misc]
