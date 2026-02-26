# rag_engine

Biblioteca Python para construir sistemas RAG (Retrieval-Augmented Generation) con LangChain, completamente desacoplada de cualquier interfaz de usuario.

`rag_engine` permite conectar una base de datos vectorial con modelos de lenguaje para responder preguntas basandose en el contenido de tus documentos. Soporta multiples proveedores de LLM (Google Gemini, OpenAI) y de bases vectoriales (ChromaDB, Pinecone), con una API unificada que abstrae las diferencias entre ellos.

---

## Caracteristicas

- **Desacoplada de UI** — No depende de Streamlit, Flask ni ninguna interfaz. Se usa como cualquier otra libreria Python.
- **Multi-proveedor de LLM** — Google Gemini y OpenAI. Extensible a otros proveedores.
- **Multi-vector DB** — ChromaDB (local) y Pinecone (cloud). Cambiar entre ellos es solo cambiar la config.
- **Single-pass retrieval** — La cadena RAG retorna la respuesta y los documentos fuente en una sola invocacion, sin busquedas duplicadas.
- **Busqueda hibrida** — Combina MMR (Maximal Marginal Relevance), MultiQuery y Ensemble para maximizar la calidad del retrieval.
- **Ingesta standalone** — `DocumentIngester` permite cargar PDFs sin necesidad de instanciar el engine completo.
- **Configuracion explicita** — El usuario siempre declara que proveedor, modelos y API key usa. Sin magia oculta.

---

## Requisitos previos

- Python 3.10 o superior
- Una API key del proveedor de LLM que vayas a usar (Google o OpenAI)
- Si usas Pinecone: una API key de Pinecone y un indice creado

---

## Instalacion

### Desde GitHub

```bash
# Instalar con todas las dependencias
pip install "rag-engine[all] @ git+https://github.com/nico-mauber/rag_engine_lib.git"

# O instalar solo lo que necesites
pip install "rag-engine[google,chroma] @ git+https://github.com/nico-mauber/rag_engine_lib.git"
pip install "rag-engine[google,pinecone] @ git+https://github.com/nico-mauber/rag_engine_lib.git"
pip install "rag-engine[openai,chroma] @ git+https://github.com/nico-mauber/rag_engine_lib.git"
```

### Desde una ruta local (desarrollo)

```bash
pip install -e "C:\ruta\a\rag_engine_lib[all]"
```

### Grupos de dependencias disponibles

| Grupo      | Que instala                              |
|------------|------------------------------------------|
| `google`   | LLM y embeddings de Google Gemini        |
| `openai`   | LLM y embeddings de OpenAI               |
| `chroma`   | ChromaDB (base vectorial local)          |
| `pinecone` | Pinecone (base vectorial en la nube)     |
| `hybrid`   | Busqueda hibrida (Ensemble + MultiQuery) |
| `pdf`      | Carga de documentos PDF                  |
| `dev`      | pytest y herramientas de testing         |
| `all`      | Todo lo anterior                         |

---

## Configuracion de API keys

Las API keys se pasan siempre por codigo, a traves de la config. Vos decidis de donde las lees: un archivo `.env`, una variable de entorno, un secrets manager, o directamente hardcodeadas para pruebas.

### Ejemplo con archivo .env (recomendado)

Instalar `python-dotenv`:

```bash
pip install python-dotenv
```

Crear un archivo `.env` en la raiz de tu proyecto:

```env
GOOGLE_API_KEY=AIzaSy...
PINECONE_API_KEY=pcsk_...
```

En tu codigo:

```python
import os
from dotenv import load_dotenv
from rag_engine import (
    RAGEngine,
    ConfigRAGElements,
    VectorDBConfig,
    VectorDBType,
    LLMConfig,
    LLMProvider,
)

load_dotenv()

config = ConfigRAGElements(
    llm=LLMConfig(
        provider=LLMProvider.GOOGLE,
        embedding_model="models/gemini-embedding-001",
        query_model="models/gemini-2.5-flash",
        generation_model="models/gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
    ),
    vector_db=VectorDBConfig(
        db_type=VectorDBType.CHROMA,
        collection="langchain",
        path="./mi_chroma_db",
    ),
)
```

Los nombres de las variables en el `.env` los elegis vos. Pueden ser `GOOGLE_API_KEY`, `MI_KEY`, o lo que quieras — vos los lees con `os.getenv()` y se los pasas a la config.

**Importante:** Agrega `.env` a tu `.gitignore` para no subir las keys al repositorio.

---

## Guia rapida

### 1. Consultar documentos ya indexados (ChromaDB)

```python
import os
from dotenv import load_dotenv
from rag_engine import RAGEngine, ConfigRAGElements, VectorDBConfig, VectorDBType, LLMConfig, LLMProvider

load_dotenv()

config = ConfigRAGElements(
    llm=LLMConfig(
        provider=LLMProvider.GOOGLE,
        embedding_model="models/gemini-embedding-001",
        query_model="models/gemini-2.5-flash",
        generation_model="models/gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
    ),
    vector_db=VectorDBConfig(
        db_type=VectorDBType.CHROMA,
        collection="langchain",
        path="./mi_chroma_db",
    ),
)

engine = RAGEngine(config=config)
response = engine.query("Cual es el plazo del contrato?")

print(response.answer)

for doc in response.source_documents:
    print(f"  Fuente: {doc.source} - Pagina: {doc.page}")
```

### 2. Ingestar PDFs por primera vez

Si tenes PDFs y queres crear la base de datos vectorial desde cero:

```python
from rag_engine import ConfigRAGElements, VectorDBConfig, VectorDBType, LLMConfig, LLMProvider
from rag_engine.ingestion import DocumentIngester

config = ConfigRAGElements(
    llm=LLMConfig(
        provider=LLMProvider.GOOGLE,
        embedding_model="models/gemini-embedding-001",
        query_model="models/gemini-2.5-flash",
        generation_model="models/gemini-2.5-flash",
        api_key="tu_google_api_key",
    ),
    vector_db=VectorDBConfig(
        db_type=VectorDBType.CHROMA,
        collection="langchain",
        path="./mi_chroma_db",
    ),
)

ingester = DocumentIngester(config=config)
n_chunks = ingester.ingest_pdf_directory("./mis_pdfs")
print(f"{n_chunks} fragmentos indexados")
```

Despues de ingestar, ya podes consultar usando el ejemplo del paso 1.

### 3. Usar Pinecone en lugar de ChromaDB

Solo cambia la configuracion del `vector_db`:

```python
config = ConfigRAGElements(
    llm=LLMConfig(
        provider=LLMProvider.GOOGLE,
        embedding_model="models/gemini-embedding-001",
        query_model="models/gemini-2.5-flash",
        generation_model="models/gemini-2.5-flash",
        api_key="tu_google_api_key",
    ),
    vector_db=VectorDBConfig(
        db_type=VectorDBType.PINECONE,
        collection="langchain",
        pinecone_index_name="mi-indice",
        pinecone_api_key="tu_pinecone_api_key",
    ),
)
```

Todo lo demas (engine, ingestion, queries) funciona exactamente igual.

**Nota:** Al crear el indice en Pinecone, elegir "I'll bring my own vectors" y configurar las dimensiones segun el modelo de embeddings que uses (por ejemplo, `gemini-embedding-001` usa 3072 dimensiones, metrica `cosine`).

### 4. Usar OpenAI en lugar de Google

Solo cambia la configuracion del `llm`:

```python
config = ConfigRAGElements(
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        embedding_model="text-embedding-3-small",
        query_model="gpt-4o-mini",
        generation_model="gpt-4o",
        api_key="tu_openai_api_key",
    ),
    vector_db=VectorDBConfig(
        db_type=VectorDBType.CHROMA,
        collection="langchain",
        path="./mi_chroma_db",
    ),
)
```

---

## Referencia de configuracion

### LLMConfig

| Parametro              | Tipo         | Obligatorio | Default | Descripcion                          |
|------------------------|--------------|-------------|---------|--------------------------------------|
| `provider`             | LLMProvider  | SI          | —       | Proveedor: `GOOGLE` o `OPENAI`       |
| `embedding_model`      | str          | SI          | —       | Modelo para generar embeddings       |
| `query_model`          | str          | SI          | —       | Modelo para reformular consultas     |
| `generation_model`     | str          | SI          | —       | Modelo para generar respuestas       |
| `api_key`              | str          | SI          | —       | API key del proveedor LLM            |
| `query_temperature`    | float        | NO          | `0.0`   | Temperatura del modelo de consultas  |
| `generation_temperature` | float      | NO          | `0.0`   | Temperatura del modelo de generacion |

### VectorDBConfig

| Parametro             | Tipo         | Obligatorio | Default     | Descripcion                                |
|-----------------------|--------------|-------------|-------------|--------------------------------------------|
| `db_type`             | VectorDBType | SI          | —           | Tipo: `CHROMA` o `PINECONE`                |
| `collection`          | str          | SI          | —           | Nombre de la coleccion                     |
| `path`                | str o None   | Solo CHROMA | None        | Ruta al directorio de ChromaDB             |
| `pinecone_index_name` | str o None   | Solo PINECONE | None      | Nombre del indice en Pinecone              |
| `pinecone_api_key`    | str o None   | NO          | None        | API key de Pinecone                        |

### RetrieverConfig (todo opcional)

| Parametro              | Tipo              | Default      | Descripcion                                   |
|------------------------|-------------------|--------------|-----------------------------------------------|
| `search_type`          | SearchType        | `MMR`        | Tipo de busqueda: `MMR` o `SIMILARITY`        |
| `search_k`             | int               | `2`          | Cantidad de documentos a retornar             |
| `mmr_diversity_lambda` | float             | `0.7`        | Balance relevancia (1.0) vs diversidad (0.0)  |
| `mmr_fetch_k`          | int               | `20`         | Candidatos pre-filtro MMR                     |
| `enable_hybrid_search` | bool              | `True`       | Activar busqueda hibrida (Ensemble)           |
| `similarity_threshold` | float             | `0.70`       | Umbral minimo de similitud                    |
| `ensemble_weights`     | tuple[float,float]| `(0.7, 0.3)` | Pesos del Ensemble (MultiQuery, Similarity)  |

### SplitterConfig (todo opcional)

| Parametro       | Tipo | Default | Descripcion                          |
|-----------------|------|---------|--------------------------------------|
| `chunk_size`    | int  | `5000`  | Tamano maximo de cada fragmento      |
| `chunk_overlap` | int  | `1000`  | Superposicion entre fragmentos       |

---

## Respuesta del engine

`engine.query()` retorna un objeto `SearchResponse`:

```python
response = engine.query("mi pregunta")

response.answer              # str - la respuesta generada
response.source_documents    # list[SourceDocument] - documentos usados como contexto
response.data                # alias de source_documents
response.ok                  # bool - True si no hubo errores
response.error               # str o None - mensaje de error si fallo
```

Cada `SourceDocument` contiene:

```python
doc.content    # str - texto del fragmento
doc.source     # str - nombre del archivo fuente
doc.page       # int o str - numero de pagina
doc.metadata   # dict - metadatos completos del documento
```

---

## Estructura del proyecto

```
rag_engine_lib/
  pyproject.toml              # Definicion del paquete y dependencias
  rag_engine/
    __init__.py               # Exportaciones publicas
    config.py                 # Enums y dataclasses de configuracion
    models.py                 # SearchResponse, SourceDocument
    prompts.py                # Templates de prompts por defecto
    engine.py                 # RAGEngine (orquestador principal)
    llm_factory.py            # Creacion de LLMs y embeddings
    retriever_factory.py      # Construccion de retrievers
    chain_factory.py          # Cadena LCEL (single-pass)
    vectordb/
      __init__.py             # Factory de proveedores
      base.py                 # VectorStoreProvider (ABC)
      chroma_provider.py      # Implementacion ChromaDB
      pinecone_provider.py    # Implementacion Pinecone
    ingestion/
      __init__.py
      ingester.py             # DocumentIngester
    _internal/
      doc_formatter.py        # Formateo de documentos para el prompt
  tests/                      # Tests unitarios (con mocks, sin API keys)
  examples/                   # Ejemplos de uso
```

---

## Ejemplo completo: proyecto desde cero

```bash
# 1. Crear proyecto
mkdir mi_proyecto
cd mi_proyecto
python -m venv venv
venv\Scripts\activate

# 2. Instalar la libreria desde GitHub
pip install "rag-engine[all] @ git+https://github.com/nico-mauber/rag_engine_lib.git"

# 3. Instalar python-dotenv para cargar keys desde .env
pip install python-dotenv
```

Crear un archivo `.env`:

```env
GOOGLE_API_KEY=AIzaSy...
```

Crear `main.py`:

```python
import os
from dotenv import load_dotenv
from rag_engine import RAGEngine, ConfigRAGElements, VectorDBConfig, VectorDBType, LLMConfig, LLMProvider

load_dotenv()

config = ConfigRAGElements(
    llm=LLMConfig(
        provider=LLMProvider.GOOGLE,
        embedding_model="models/gemini-embedding-001",
        query_model="models/gemini-2.5-flash",
        generation_model="models/gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
    ),
    vector_db=VectorDBConfig(
        db_type=VectorDBType.CHROMA,
        collection="langchain",
        path="./mi_chroma_db",
    ),
)

engine = RAGEngine(config=config)
response = engine.query("Tu pregunta aqui")
print(response.answer)
```

Ejecutar:

```bash
python main.py
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Los tests usan mocks y no requieren API keys ni bases de datos reales.

---

## Notas de diseno

- **RAGEngine es inmutable** — Una vez creado, no se puede reconfigurar. Para cambiar la configuracion, se crea una nueva instancia.
- **Configuracion explicita** — No hay valores magicos. El usuario siempre declara su proveedor, modelos y API key. Si falta algo obligatorio, da error inmediato.
- **Dependency injection** — Se pueden inyectar objetos LLM, embeddings o vectorstore pre-configurados en el constructor de `RAGEngine` y `DocumentIngester`, util para testing o casos avanzados.
- **Lazy imports** — Las dependencias de cada proveedor (Google, OpenAI, Chroma, Pinecone) se importan solo cuando se necesitan, evitando errores si no estan instaladas.
