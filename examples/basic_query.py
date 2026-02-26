"""Ejemplo basico: ejecutar una consulta RAG contra ChromaDB existente."""

from rag_engine import ConfigRAGElements, RAGEngine, VectorDBConfig

# Configurar apuntando a un ChromaDB existente
config = ConfigRAGElements(
    vector_db=VectorDBConfig(path="./chroma_db"),
)

engine = RAGEngine(config=config)

# Ejecutar consulta
response = engine.query("¿Donde se encuentra el local del contrato de Maria Jimenez Campos?")

print("Respuesta:", response.answer)
print()
for doc in response.source_documents:
    print(f"  Fuente: {doc.source} - Pagina: {doc.page}")
    print(f"  {doc.content[:200]}...")
    print()
