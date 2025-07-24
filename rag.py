import os
import shutil
from datetime import datetime

import pandas as pd
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# === Config ===
DOCS_FOLDER = "docs"
TEMP_FOLDER = "temp"
PERSIST_DIR = "./rag_data"
os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    ".pdf": PyMuPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": UnstructuredWordDocumentLoader,
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

# === Streamlit UI ===
st.title("📄 Multi-file RAG App with History, Model & Embedding Switch")

# Upload new files
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

# File list with remove button
st.sidebar.markdown("📂 **Loaded Files**")
files = [
    f for f in os.listdir(DOCS_FOLDER)
    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
]
if not files:
    st.sidebar.info("No documents found.")
else:
    for f in files:
        col1, col2 = st.sidebar.columns([5, 1])
        col1.markdown(f"{f}")
        if col2.button("x", key=f"delete_{f}", help=f"Remove {f} (move to temp)"):
            source = os.path.join(DOCS_FOLDER, f)
            destination = os.path.join(TEMP_FOLDER, f)
            shutil.move(source, destination)
            st.sidebar.success(f"Moved {f} to {TEMP_FOLDER}/")
            st.cache_resource.clear()
            st.rerun()


# Model and embedding settings
st.sidebar.header("🤖 LLM Settings")
selected_model = st.sidebar.selectbox("Choose Ollama Model", AVAILABLE_MODELS, index=0)
embedding_model = st.sidebar.selectbox("Embedding Model", EMBEDDING_MODELS, index=0)

# Chunking settings
st.sidebar.header("⚙️ Chunking Settings")
use_case = st.selectbox("Use Case", list(USE_CASES.keys()))
default_chunk_size, default_chunk_overlap = USE_CASES[use_case]
chunk_size = st.slider("Chunk Size", 100, 2000, default_chunk_size, 100)
chunk_overlap = st.slider("Chunk Overlap", 0, 500, default_chunk_overlap, 10)

# Rebuild logic
if "clear_requested" not in st.session_state:
    st.session_state.clear_requested = False

if st.sidebar.button("🔄 Force Rebuild Embeddings"):
    st.session_state.clear_requested = True
    st.rerun()

if st.session_state.clear_requested:
    if os.path.exists(PERSIST_DIR):
        try:
            shutil.rmtree(PERSIST_DIR)
            st.success("✅ Cleared existing embeddings.")
        except Exception as e:
            st.error(f"Failed to clear embeddings: {e}")
    st.cache_resource.clear()
    st.session_state.clear_requested = False
    st.rerun()

# === LangChain QA Pipeline ===
@st.cache_resource
def build_qa_chain(chunk_size, chunk_overlap, model_name, embedding_choice):
    all_docs = []
    for filename in os.listdir(DOCS_FOLDER):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            loader_cls = SUPPORTED_EXTENSIONS[ext]
            file_path = os.path.join(DOCS_FOLDER, filename)
            loader = loader_cls(file_path)
            docs = loader.load()
            all_docs.extend(docs)

    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(all_docs)

    if embedding_choice == "ollama":
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    elif embedding_choice == "huggingface":
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        st.error("Invalid embedding model selected.")
        return

    db = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)

    llm = OllamaLLM(model=model_name)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(), return_source_documents=True)
    return qa

qa_chain = build_qa_chain(chunk_size, chunk_overlap, selected_model, embedding_model)

# Q&A Interaction
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

bottom_input = st.empty()
with bottom_input.container():
    st.subheader("💬 Ask Your Question")
    query = st.text_input("What would you like to know from the documents?", key="user_input")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        sources = [doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.qa_history.append({
        "timestamp": now,
        "question": query,
        "answer": answer,
        "model": selected_model,
        "sources": sources
    })

# Display Full History
if st.session_state.qa_history:
    st.subheader("📜 Full Q&A History")
    for chat in st.session_state.qa_history:
        st.markdown(f"**🕑 {chat['timestamp']} — Model: `{chat['model']}`**")
        st.markdown(f"**Q:** {chat['question']}")
        st.markdown(f"**A:** {chat['answer']}")
        if chat.get("sources"):
            st.markdown("📂 **Sources used:**")
            for src in chat["sources"]:
                st.markdown(f"- `{src}`")
        st.markdown("---")

    df = pd.DataFrame(st.session_state.qa_history)
    txt = "\n\n".join(
        f"[{r['timestamp']}] ({r['model']})\nQ: {r['question']}\nA: {r['answer']}"
        for r in st.session_state.qa_history
    )
    st.download_button("⬇️ Download as TXT", txt, file_name="qa_history.txt")
    st.download_button("⬇️ Download as CSV", df.to_csv(index=False), file_name="qa_history.csv")
