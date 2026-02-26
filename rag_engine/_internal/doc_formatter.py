"""Utilidad interna para formatear documentos recuperados como contexto."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


def format_docs(docs: list[Document]) -> str:
    """Convierte una lista de Documents de LangChain en texto formateado para el prompt.

    Cada fragmento incluye un encabezado numerado, la fuente y pagina (si existen),
    y el contenido textual.
    """
    formatted: list[str] = []

    for i, doc in enumerate(docs, 1):
        header = f"[Fragmento {i}]"

        if doc.metadata:
            if "source" in doc.metadata:
                source = os.path.basename(doc.metadata["source"])
                header += f" - Fuente: {source}"
            if "page" in doc.metadata:
                header += f" - Pagina: {doc.metadata['page']}"

        content = doc.page_content.strip()
        formatted.append(f"{header}\n{content}")

    return "\n\n".join(formatted)
