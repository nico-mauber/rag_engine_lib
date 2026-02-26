"""Ejemplo: app Streamlit usando rag_engine como biblioteca."""

import streamlit as st

from rag_engine import ConfigRAGElements, RAGEngine, VectorDBConfig

# -- Configuracion -----------------------------------------------------------

st.set_page_config(page_title="Sistema RAG - Asistente Legal", page_icon="⚖️", layout="wide")


@st.cache_resource
def _get_engine() -> RAGEngine:
    config = ConfigRAGElements(
        vector_db=VectorDBConfig(path="./chroma_db"),
    )
    return RAGEngine(config=config)


engine = _get_engine()

# -- UI -----------------------------------------------------------------------

st.title("⚖️ Sistema RAG - Asistente Legal")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("📋 Información del Sistema")
    info = engine.get_retriever_info()
    st.markdown("**🔍 Retriever:**")
    st.info(f"Tipo: {info['tipo']}")
    st.markdown("**🤖 Modelos:**")
    st.info(f"Query: {engine.config.llm.query_model}\nGeneración: {engine.config.llm.generation_model}")
    st.divider()
    if st.button("🗑️ Limpiar Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Layout principal
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with col2:
    st.markdown("### 📄 Documentos Relevantes")
    if st.session_state.messages:
        last = st.session_state.messages[-1]
        if last["role"] == "assistant" and "docs" in last:
            for doc in last["docs"]:
                with st.expander(f"📄 {doc['source']} - Pag. {doc['page']}", expanded=False):
                    st.text(doc["content"])

# Chat input
if prompt := st.chat_input("Escribe tu consulta sobre contratos de arrendamiento..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("🔍 Analizando..."):
        response = engine.query(prompt)
        docs_info = [
            {"source": d.source, "page": d.page, "content": d.content}
            for d in response.source_documents
        ]
        st.session_state.messages.append({"role": "assistant", "content": response.answer, "docs": docs_info})

    st.rerun()

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: #666;'>🏛️ Asistente Legal con RAG Engine</div>",
    unsafe_allow_html=True,
)
