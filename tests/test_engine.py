"""Tests para RAGEngine con mocks (sin API keys ni ChromaDB real)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag_engine.config import (
    ConfigRAGElements,
    LLMConfig,
    LLMProvider,
    RetrieverConfig,
    VectorDBConfig,
    VectorDBType,
)
from rag_engine.engine import RAGEngine
from rag_engine.models import SearchResponse


# -- Helpers --

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
    defaults = dict(db_type=VectorDBType.CHROMA, collection="langchain", path="/tmp/test")
    defaults.update(overrides)
    return VectorDBConfig(**defaults)


def _config(**overrides):
    defaults = dict(llm=_llm(), vector_db=_vdb())
    defaults.update(overrides)
    return ConfigRAGElements(**defaults)


@pytest.fixture
def mock_embeddings():
    emb = MagicMock()
    emb.embed_documents.return_value = [[0.1, 0.2]]
    emb.embed_query.return_value = [0.1, 0.2]
    return emb


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Respuesta generada")
    return llm


@pytest.fixture
def mock_vectorstore():
    vs = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = [
        Document(page_content="contenido legal", metadata={"source": "/path/contrato.pdf", "page": 1}),
    ]
    vs.as_retriever.return_value = retriever
    return vs


class TestRAGEngineInit:
    @patch("rag_engine.engine.build_rag_chain")
    @patch("rag_engine.engine.build_retriever")
    def test_init_with_injected_deps(
        self, mock_build_retriever, mock_build_chain, mock_embeddings, mock_llm, mock_vectorstore
    ):
        mock_build_retriever.return_value = MagicMock()
        mock_build_chain.return_value = MagicMock()

        config = _config()
        engine = RAGEngine(
            config=config,
            embeddings=mock_embeddings,
            llm_generation=mock_llm,
            llm_queries=mock_llm,
            vectorstore=mock_vectorstore,
        )

        assert engine.config is config
        assert engine.vectorstore is mock_vectorstore
        assert engine.embeddings is mock_embeddings
        mock_build_retriever.assert_called_once()
        mock_build_chain.assert_called_once()


class TestRAGEngineQuery:
    @patch("rag_engine.engine.build_rag_chain")
    @patch("rag_engine.engine.build_retriever")
    def test_query_returns_search_response(
        self, mock_build_retriever, mock_build_chain, mock_embeddings, mock_llm, mock_vectorstore
    ):
        mock_retriever = MagicMock()
        mock_build_retriever.return_value = mock_retriever

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "answer": "La propiedad esta en Montevideo",
            "source_documents": [
                Document(page_content="contenido", metadata={"source": "contrato.pdf", "page": 2}),
            ],
        }
        mock_build_chain.return_value = mock_chain

        engine = RAGEngine(
            config=_config(),
            embeddings=mock_embeddings,
            llm_generation=mock_llm,
            llm_queries=mock_llm,
            vectorstore=mock_vectorstore,
        )

        response = engine.query("Donde esta la propiedad?")

        assert isinstance(response, SearchResponse)
        assert response.ok is True
        assert "Montevideo" in response.answer
        assert len(response.source_documents) == 1
        assert response.source_documents[0].source == "contrato.pdf"
        assert response.source_documents[0].page == 2

    @patch("rag_engine.engine.build_rag_chain")
    @patch("rag_engine.engine.build_retriever")
    def test_query_handles_error(
        self, mock_build_retriever, mock_build_chain, mock_embeddings, mock_llm, mock_vectorstore
    ):
        mock_build_retriever.return_value = MagicMock()

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("API down")
        mock_build_chain.return_value = mock_chain

        engine = RAGEngine(
            config=_config(),
            embeddings=mock_embeddings,
            llm_generation=mock_llm,
            llm_queries=mock_llm,
            vectorstore=mock_vectorstore,
        )

        response = engine.query("test question")

        assert response.ok is False
        assert "API down" in response.error
        assert response.answer == ""


class TestGetRetrieverInfo:
    @patch("rag_engine.engine.build_rag_chain")
    @patch("rag_engine.engine.build_retriever")
    def test_info_hybrid(
        self, mock_build_retriever, mock_build_chain, mock_embeddings, mock_llm, mock_vectorstore
    ):
        mock_build_retriever.return_value = MagicMock()
        mock_build_chain.return_value = MagicMock()

        engine = RAGEngine(
            config=_config(retriever=RetrieverConfig(enable_hybrid_search=True)),
            embeddings=mock_embeddings,
            llm_generation=mock_llm,
            llm_queries=mock_llm,
            vectorstore=mock_vectorstore,
        )

        info = engine.get_retriever_info()
        assert "Hybrid" in info["tipo"]
        assert info["documentos"] == 2

    @patch("rag_engine.engine.build_rag_chain")
    @patch("rag_engine.engine.build_retriever")
    def test_info_no_hybrid(
        self, mock_build_retriever, mock_build_chain, mock_embeddings, mock_llm, mock_vectorstore
    ):
        mock_build_retriever.return_value = MagicMock()
        mock_build_chain.return_value = MagicMock()

        engine = RAGEngine(
            config=_config(retriever=RetrieverConfig(enable_hybrid_search=False)),
            embeddings=mock_embeddings,
            llm_generation=mock_llm,
            llm_queries=mock_llm,
            vectorstore=mock_vectorstore,
        )

        info = engine.get_retriever_info()
        assert "Hybrid" not in info["tipo"]
        assert info["umbral"] == "N/A"
