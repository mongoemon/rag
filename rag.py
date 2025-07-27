# rag.py
import os
import shutil
from datetime import datetime
import pandas as pd
import streamlit as st
from rag_loader import build_qa_chain_with_warnings as build_qa_chain

# === Config ===
DOCS_FOLDER = "docs"
TEMP_FOLDER = "temp"
PERSIST_DIR = "./rag_data"
os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF",
    ".txt": "Text",
    ".md": "Markdown",
    ".docx": "Word",
}

USE_CASES = {
    "General Purpose": (500, 50),
    "Technical Docs / API": (1000, 150),
    "PDF Reports / eBooks": (1200, 200),
    "Code Files / Dev Docs": (300, 50),
    "Legal / Contracts": (1500, 300),
    "Short Notes / FAQs": (400, 40),
}

AVAILABLE_MODELS = ["mistral", "llama3", "phi3", "codellama", "gemma", "dolphin-mixtral"]
EMBEDDING_MODELS = ["ollama", "huggingface"]

OLLAMA_MODEL_TOOLTIPS = """
- **mistral**: Fast and efficient general-purpose LLM.
- **llama3**: Meta's latest model with strong reasoning and general capabilities.
- **phi3**: Lightweight Microsoft model optimized for smaller use cases.
- **codellama**: Code-specialized variant of LLaMA, great for code understanding and generation.
- **gemma**: Google's open model designed for general-purpose LLM tasks.
- **dolphin-mixtral**: Chat-tuned Mixtral blend optimized for interactive Q&A.
"""

EMBEDDING_MODEL_TOOLTIPS = """
- **ollama**: Uses Ollama local embeddings—fast and private.
- **huggingface**: Leverages HuggingFace hosted models—flexible but requires internet and API compatibility.
"""

st.title("📄 Multi-file RAG App with History, Model & Embedding Switch")

# Upload Section
st.sidebar.header("📤 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, TXT, MD, or DOCX files",
    type=list(SUPPORTED_EXTENSIONS.keys()),
    accept_multiple_files=True,
)
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(DOCS_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s).")
    st.cache_resource.clear()
    st.rerun()

# Display Loaded Files
st.sidebar.markdown("📂 **Loaded Files**")
files = [f for f in os.listdir(DOCS_FOLDER) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
if not files:
    st.sidebar.info("No documents found.")
else:
    for f in files:
        col1, col2 = st.sidebar.columns([5, 1])
        col1.markdown(f"{f}")
        if col2.button("x", key=f"delete_{f}", help=f"Remove {f}"):
            shutil.move(os.path.join(DOCS_FOLDER, f), os.path.join(TEMP_FOLDER, f))
            st.sidebar.success(f"Moved {f} to temp/")
            st.cache_resource.clear()
            st.rerun()

# Sidebar Settings
st.sidebar.header("🤖 LLM Settings")
selected_model = st.sidebar.selectbox(
    "Choose Ollama Model",
    AVAILABLE_MODELS,
    index=0,
    help=OLLAMA_MODEL_TOOLTIPS.strip()
)
embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    EMBEDDING_MODELS,
    index=0,
    help=EMBEDDING_MODEL_TOOLTIPS.strip()
)

st.sidebar.header("⚙️ Chunking Settings")
use_case = st.selectbox("Use Case", list(USE_CASES.keys()))
default_chunk_size, default_chunk_overlap = USE_CASES[use_case]
chunk_size = st.slider(
    "Chunk Size", 100, 2000, default_chunk_size, 100,
    help="Number of characters in each chunk. Smaller values give better precision but may lose context. Larger values preserve more context but may cause the model to miss specific details."
)
chunk_overlap = st.slider(
    "Chunk Overlap", 0, 500, default_chunk_overlap, 10,
    help="Number of overlapping characters between chunks. More overlap ensures smoother transitions across chunks but increases redundancy and memory usage. Less overlap improves efficiency but may reduce answer quality for questions near chunk boundaries."
)

if "clear_requested" not in st.session_state:
    st.session_state.clear_requested = False
if st.sidebar.button("🔄 Force Rebuild Embeddings"):
    st.session_state.clear_requested = True
    st.rerun()
if st.session_state.clear_requested:
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        st.success("✅ Cleared existing embeddings.")
    st.cache_resource.clear()
    st.session_state.clear_requested = False
    st.rerun()

with st.spinner("📚 Building document index and embeddings..."):
    qa_chain, warnings = build_qa_chain(DOCS_FOLDER, chunk_size, chunk_overlap, selected_model, embedding_model, PERSIST_DIR)

if warnings:
    with st.expander("⚠️ Chunking Warnings"):
        for w in warnings:
            st.markdown(f"- {w}")


# Question & Answer Section
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

bottom_input = st.empty()
with st.form("question_form", clear_on_submit=False):
    st.subheader("💬 Ask Your Question")
    query = st.text_area(
        "What would you like to know from the documents?",
        key="user_input",
        height=100,
        help="Press Ctrl+Enter to insert a newline. Press Enter to submit."
    )

    submitted = st.form_submit_button("Submit", type="primary")

if submitted and query and qa_chain:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        sources = [
            f"{doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))} ({doc.metadata.get('line_range', 'L?')})"
            for doc in result.get("source_documents", [])
        ]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.qa_history.append({
        "timestamp": now,
        "question": query,
        "answer": answer,
        "model": selected_model,
        "sources": sources
    })

# Display Q&A History
if st.session_state.qa_history:
    st.subheader("📜 Full Q&A History")
    for chat in st.session_state.qa_history:
        st.markdown(f"**🕑 {chat['timestamp']} — Model: `{chat['model']}`**")
        st.markdown(f"**Q:** {chat['question']}")
        st.markdown(f"**A:** {chat['answer']}")
        if chat.get("sources"):
            st.markdown("📂 **Sources used:**")
            for src in sorted(set(chat["sources"])):
                st.markdown(f"- `{src}`")
        st.markdown("---")

    df = pd.DataFrame(st.session_state.qa_history)
    txt = "\n\n".join(f"[{r['timestamp']}] ({r['model']})\nQ: {r['question']}\nA: {r['answer']}" for r in st.session_state.qa_history)
    st.download_button("⬇️ Download as TXT", txt, file_name="qa_history.txt")
    st.download_button("⬇️ Download as CSV", df.to_csv(index=False), file_name="qa_history.csv")
