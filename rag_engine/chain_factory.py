"""Factory para construir la cadena LCEL que retorna answer + docs en un solo paso."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from rag_engine._internal.doc_formatter import format_docs
from rag_engine.prompts import RAG_TEMPLATE

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


def build_rag_chain(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    *,
    rag_template: str | None = None,
) -> Runnable:
    """Construye la cadena RAG que retorna ``{"answer": str, "source_documents": list[Document]}``.

    Se usa ``RunnablePassthrough.assign()`` para capturar los documentos intermedios
    y evitar un doble retrieval.
    """
    template = rag_template or RAG_TEMPLATE
    prompt = PromptTemplate.from_template(template)

    # Paso 1: recuperar documentos y guardarlos en "context" (lista de Documents)
    retrieval_step = RunnablePassthrough.assign(
        context=lambda x: retriever.invoke(x["question"])
    )

    # Paso 2: formatear contexto, generar respuesta, y preservar docs originales
    def _generate(inputs: dict[str, Any]) -> dict[str, Any]:
        docs = inputs["context"]
        formatted = format_docs(docs)
        prompt_value = prompt.invoke({"context": formatted, "question": inputs["question"]})
        ai_message = llm.invoke(prompt_value)
        answer = StrOutputParser().invoke(ai_message)
        return {"answer": answer, "source_documents": docs}

    chain: Runnable = retrieval_step | RunnableLambda(_generate)

    logger.info("Cadena RAG construida (single-pass con docs)")
    return chain
