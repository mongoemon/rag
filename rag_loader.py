# rag_loader.py
import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)

from langchain_community.document_loaders.text import TextLoader as RawTextLoader
from langchain_core.documents import Document

class SafeTextLoader(TextLoader):
    def load(self):
        try:
            return super().load()
        except Exception:
            try:
                with open(self.file_path, "r", encoding="utf-8-sig") as f:
                    text = f.read()
                return [Document(page_content=text, metadata={"source": self.file_path})]
            except Exception as e:
                raise RuntimeError(f"TextLoader failed with fallback: {e}")

SUPPORTED_EXTENSIONS = {
    ".pdf": PyMuPDFLoader,
    ".txt": SafeTextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": UnstructuredWordDocumentLoader,
}

def build_qa_chain_with_warnings(docs_folder, chunk_size, chunk_overlap, model_name, embedding_choice, persist_dir):
    warnings = []
    all_docs = []
    for filename in os.listdir(docs_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            file_path = os.path.join(docs_folder, filename)
            loader_cls = SUPPORTED_EXTENSIONS[ext]
            try:
                if os.path.getsize(file_path) == 0:
                    st.warning(f"⚠️ Skipped empty file: {filename}")
                    continue
                loader = loader_cls(file_path)
                docs = loader.load()
                for doc in docs:
                    lines = doc.page_content.splitlines()
                    if lines:
                        doc.metadata["source_text"] = "\n".join(lines)
                        doc.metadata["source_file"] = filename
                all_docs.extend(docs)
            except Exception as e:
                st.warning(f"⚠️ Failed to load {filename}: {e}")
                continue

    if not all_docs:
        st.error("❌ No valid documents to process. Please upload supported, non-empty files.")
        return None, warnings

    splitter = CharacterTextSplitter(chunk_size=max(chunk_size, 1000), chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(all_docs)

    for chunk in chunks:
        source_text = chunk.metadata.get("source_text")
        if source_text:
            content = chunk.page_content.strip()
            start = source_text.find(content)
            if start != -1:
                prefix = source_text[:start].splitlines()
                line_start = len(prefix) + 1
                line_end = line_start + len(content.splitlines()) - 1
                chunk.metadata["line_range"] = f"L{line_start}-L{line_end}"
        if "line_range" not in chunk.metadata:
            chunk.metadata["line_range"] = "Unknown"
        chunk.metadata.pop("source_text", None)

        actual_size = len(chunk.page_content)
        if actual_size > chunk_size:
            warnings.append(f"Created a chunk of size {actual_size}, which is longer than the specified {chunk_size}.")

    if not chunks:
        st.error("❌ Document splitting resulted in no chunks. Try reducing chunk size or check content.")
        return None, warnings

    if embedding_choice == "ollama":
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    elif embedding_choice == "huggingface":
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        st.error("Invalid embedding model selected.")
        return None, warnings

    db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    llm = OllamaLLM(model=model_name)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True
    )
    return qa, warnings
