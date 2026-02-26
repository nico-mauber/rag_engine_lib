# rag_engine

Biblioteca Python para construir sistemas RAG (Retrieval-Augmented Generation) con LangChain, completamente desacoplada de cualquier interfaz de usuario.

`rag_engine` permite conectar una base de datos vectorial con modelos de lenguaje para responder preguntas basandose en el contenido de tus documentos. Soporta multiples proveedores de LLM (Google Gemini, OpenAI) y de bases vectoriales (ChromaDB, Pinecone), con una API unificada que abstrae las diferencias entre ellos.

---

## Caracteristicas

- **Desacoplada de UI** — No depende de Streamlit, Flask ni ninguna interfaz. Se usa como cualquier otra libreria Python.
- **Multi-proveedor de LLM** — Google Gemini y OpenAI. Extensible a otros proveedores.
- **Multi-vector DB** — ChromaDB (local) y Pinecone (cloud). Cambiar entre ellos es solo una linea de configuracion.
- **Single-pass retrieval** — La cadena RAG retorna la respuesta y los documentos fuente en una sola invocacion, sin busquedas duplicadas.
- **Busqueda hibrida** — Combina MMR (Maximal Marginal Relevance), MultiQuery y Ensemble para maximizar la calidad del retrieval.
- **Ingesta standalone** — `DocumentIngester` permite cargar PDFs sin necesidad de instanciar el engine completo.
- **Configuracion compuesta** — Sub-dataclasses para LLM, vector DB, retriever y splitter. Sin god objects.
- **API keys flexibles** — Se pueden pasar por codigo (desde un `.env`, base de datos, etc.) o dejar que se resuelvan automaticamente desde variables de entorno.

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

Hay dos formas de configurar las API keys. Usa la que prefieras.

### Opcion A: Pasar las keys por codigo

La forma recomendada. Te permite cargar las keys desde un archivo `.env`, una base de datos, un secrets manager, o como quieras. La libreria las recibe y las usa directamente.

```python
import os
from dotenv import load_dotenv  # pip install python-dotenv

load_dotenv()  # carga variables desde un archivo .env

config = ConfigRAGElements(
    llm=LLMConfig(
        provider=LLMProvider.GOOGLE,
        api_key=os.getenv("MI_GOOGLE_KEY"),
    ),
    vector_db=VectorDBConfig(
        db_type=VectorDBType.PINECONE,
        pinecone_index_name="mi-indice",
        pinecone_api_key=os.getenv("MI_PINECONE_KEY"),
    ),
)
```

Con un archivo `.env` en la raiz de tu proyecto:

```env
MI_GOOGLE_KEY=AIzaSy...
MI_PINECONE_KEY=pcsk_...
MI_OPENAI_KEY=sk-...
```

**Importante:** Agrega `.env` a tu `.gitignore` para no subir las keys al repositorio.

### Opcion B: Variables de entorno del sistema

Si no pasas `api_key` en la config, la libreria deja que LangChain las busque automaticamente en las variables de entorno estandar.

**Windows (cmd):**

```bash
set GOOGLE_API_KEY=tu_google_api_key
set OPENAI_API_KEY=tu_openai_api_key
set PINECONE_API_KEY=tu_pinecone_api_key
```

**Windows (permanente, sobrevive al cerrar la terminal):**

```bash
setx GOOGLE_API_KEY "tu_google_api_key"
```

**Linux / macOS:**

```bash
export GOOGLE_API_KEY=tu_google_api_key
export OPENAI_API_KEY=tu_openai_api_key
export PINECONE_API_KEY=tu_pinecone_api_key
```

Solo necesitas configurar las keys de los proveedores que vayas a usar.

### Resumen

| Parametro              | Variable de entorno     | Donde se configura |
|------------------------|-------------------------|--------------------|
| `LLMConfig.api_key`    | `GOOGLE_API_KEY`        | LLM y embeddings de Google |
| `LLMConfig.api_key`    | `OPENAI_API_KEY`        | LLM y embeddings de OpenAI |
| `VectorDBConfig.pinecone_api_key` | `PINECONE_API_KEY` | Conexion a Pinecone |

Si pasas `api_key` en la config, tiene prioridad sobre la variable de entorno.

---

## Guia rapida

### 1. Consultar documentos ya indexados

Si ya tenes una base de datos vectorial con documentos cargados:

```python
from rag_engine import RAGEngine, ConfigRAGElements, VectorDBConfig, LLMConfig, LLMProvider

config = ConfigRAGElements(
    llm=LLMConfig(
        provider=LLMProvider.GOOGLE,
        embedding_model="models/gemini-embedding-001",
        query_model="models/gemini-2.5-flash",
        generation_model="models/gemini-2.5-flash",
    ),
    vector_db=VectorDBConfig(
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
from rag_engine import ConfigRAGElements, VectorDBConfig
from rag_engine.ingestion import DocumentIngester

config = ConfigRAGElements(
    vector_db=VectorDBConfig(path="./mi_chroma_db"),
)

ingester = DocumentIngester(config=config)
n_chunks = ingester.ingest_pdf_directory("./mis_pdfs")
print(f"{n_chunks} fragmentos indexados")
```

Despues de ingestar, ya podes consultar usando el ejemplo del paso 1.

### 3. Usar Pinecone en lugar de ChromaDB

Solo cambia la configuracion del vector_db:

```python
from rag_engine import ConfigRAGElements, VectorDBConfig, VectorDBType

config = ConfigRAGElements(
    vector_db=VectorDBConfig(
        db_type=VectorDBType.PINECONE,
        pinecone_index_name="mi-indice",
    ),
)
```

Todo lo demas (engine, ingestion, queries) funciona exactamente igual.

**Nota:** Al crear el indice en Pinecone, elegir "I'll bring my own vectors" y configurar las dimensiones segun el modelo de embeddings que uses (por ejemplo, `gemini-embedding-001` usa 3072 dimensiones, metrica `cosine`).

### 4. Usar OpenAI en lugar de Google

Solo cambia la configuracion del LLM:

```python
from rag_engine import LLMConfig, LLMProvider

config = ConfigRAGElements(
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        embedding_model="text-embedding-3-small",
        query_model="gpt-4o-mini",
        generation_model="gpt-4o",
    ),
    vector_db=VectorDBConfig(path="./mi_chroma_db"),
)
```

---

## Referencia de configuracion

La configuracion se compone de sub-dataclasses independientes:

### LLMConfig

| Parametro              | Tipo         | Default                        | Descripcion                          |
|------------------------|--------------|--------------------------------|--------------------------------------|
| `provider`             | LLMProvider  | `GOOGLE`                       | Proveedor: `GOOGLE` o `OPENAI`       |
| `embedding_model`      | str          | `models/gemini-embedding-001`  | Modelo para generar embeddings       |
| `query_model`          | str          | `models/gemini-2.5-flash`      | Modelo para reformular consultas     |
| `generation_model`     | str          | `models/gemini-2.5-flash`      | Modelo para generar respuestas       |
| `query_temperature`    | float        | `0.0`                          | Temperatura del modelo de consultas  |
| `generation_temperature` | float      | `0.0`                          | Temperatura del modelo de generacion |
| `api_key`              | str o None   | None                           | API key del proveedor LLM. Si es None, se busca en variables de entorno |

### VectorDBConfig

| Parametro             | Tipo         | Default      | Descripcion                                |
|-----------------------|--------------|--------------|--------------------------------------------|
| `db_type`             | VectorDBType | `CHROMA`     | Tipo: `CHROMA` o `PINECONE`                |
| `path`                | str o None   | None         | Ruta al directorio de ChromaDB (requerido para CHROMA) |
| `collection`          | str          | `langchain`  | Nombre de la coleccion en ChromaDB         |
| `pinecone_index_name` | str o None   | None         | Nombre del indice en Pinecone (requerido para PINECONE) |
| `pinecone_api_key`    | str o None   | None         | API key de Pinecone. Si es None, se busca en variables de entorno |

### RetrieverConfig

| Parametro              | Tipo              | Default      | Descripcion                                   |
|------------------------|-------------------|--------------|-----------------------------------------------|
| `search_type`          | SearchType        | `MMR`        | Tipo de busqueda: `MMR` o `SIMILARITY`        |
| `search_k`             | int               | `2`          | Cantidad de documentos a retornar             |
| `mmr_diversity_lambda` | float             | `0.7`        | Balance relevancia (1.0) vs diversidad (0.0)  |
| `mmr_fetch_k`          | int               | `20`         | Candidatos pre-filtro MMR                     |
| `enable_hybrid_search` | bool              | `True`       | Activar busqueda hibrida (Ensemble)           |
| `similarity_threshold` | float             | `0.70`       | Umbral minimo de similitud                    |
| `ensemble_weights`     | tuple[float,float]| `(0.7, 0.3)` | Pesos del Ensemble (MultiQuery, Similarity)  |

### SplitterConfig

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

# 3. (Opcional) Instalar python-dotenv para cargar keys desde .env
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
from rag_engine import RAGEngine, ConfigRAGElements, VectorDBConfig, LLMConfig, LLMProvider

load_dotenv()

config = ConfigRAGElements(
    llm=LLMConfig(
        provider=LLMProvider.GOOGLE,
        api_key=os.getenv("GOOGLE_API_KEY"),
    ),
    vector_db=VectorDBConfig(path="./mi_chroma_db"),
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

- **RAGEngine es inmutable** — Una vez creado, no se puede reconfigurar. Para cambiar la configuracion, se crea una nueva instancia. Esto simplifica el estado interno y evita bugs de reconfiguracion parcial.
- **Dependency injection** — Se pueden inyectar objetos LLM, embeddings o vectorstore pre-configurados en el constructor de `RAGEngine` y `DocumentIngester`, util para testing o casos avanzados.
- **Lazy imports** — Las dependencias de cada proveedor (Google, OpenAI, Chroma, Pinecone) se importan solo cuando se necesitan, evitando errores si no estan instaladas.
