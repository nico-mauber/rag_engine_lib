"""Modelos de datos para las respuestas del engine RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceDocument:
    """Documento fuente recuperado por el retriever."""

    content: str
    source: str = "desconocida"
    page: int | str = "N/A"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """Respuesta completa de una consulta RAG.

    Atributos:
        answer: Texto generado por el LLM.
        source_documents: Documentos utilizados como contexto.
        error: Mensaje de error si la consulta fallo (answer estara vacio).
    """

    answer: str
    source_documents: list[SourceDocument] = field(default_factory=list)
    error: str | None = None

    @property
    def data(self) -> list[SourceDocument]:
        """Alias de conveniencia para source_documents."""
        return self.source_documents

    @property
    def ok(self) -> bool:
        return self.error is None
