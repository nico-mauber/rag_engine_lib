"""Tests para el modulo de modelos."""

from rag_engine.models import SearchResponse, SourceDocument


class TestSourceDocument:
    def test_defaults(self):
        doc = SourceDocument(content="texto")
        assert doc.content == "texto"
        assert doc.source == "desconocida"
        assert doc.page == "N/A"
        assert doc.metadata == {}

    def test_custom_values(self):
        doc = SourceDocument(
            content="contenido legal",
            source="contrato.pdf",
            page=3,
            metadata={"key": "val"},
        )
        assert doc.source == "contrato.pdf"
        assert doc.page == 3


class TestSearchResponse:
    def test_ok_response(self):
        docs = [SourceDocument(content="a"), SourceDocument(content="b")]
        resp = SearchResponse(answer="La respuesta es...", source_documents=docs)
        assert resp.ok is True
        assert resp.error is None
        assert resp.answer == "La respuesta es..."
        assert len(resp.source_documents) == 2
        assert resp.data is resp.source_documents

    def test_error_response(self):
        resp = SearchResponse(answer="", error="algo salio mal")
        assert resp.ok is False
        assert resp.error == "algo salio mal"
        assert resp.source_documents == []

    def test_empty_defaults(self):
        resp = SearchResponse(answer="ok")
        assert resp.source_documents == []
        assert resp.ok is True
