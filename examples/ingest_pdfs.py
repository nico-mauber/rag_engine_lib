"""Ejemplo: ingestar PDFs en ChromaDB usando DocumentIngester standalone."""

from rag_engine import ConfigRAGElements, VectorDBConfig
from rag_engine.ingestion import DocumentIngester

config = ConfigRAGElements(
    vector_db=VectorDBConfig(path="./chroma_db"),
)

ingester = DocumentIngester(config=config)
num_chunks = ingester.ingest_pdf_directory("./contratos")

print(f"Se ingresaron {num_chunks} chunks al vector store.")
