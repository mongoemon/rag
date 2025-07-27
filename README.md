# 🧠 Multi-file RAG App with Streamlit + Ollama + LangChain

This is a powerful RAG (Retrieval-Augmented Generation) web app that allows you to upload documents (PDF, TXT, MD, DOCX), ask questions about them, and get AI-generated answers using **local Ollama LLMs** and **LangChain**.

---

## 🚀 Features

- ✅ Upload multiple `.pdf`, `.txt`, `.md`, `.docx` documents
- ✅ Dynamically select a use case to adjust chunk size and overlap
- ✅ Choose from multiple Ollama models (e.g., `mistral`, `llama3`, `phi3`)
- ✅ Delete documents via the interface
- ✅ Ask questions and get instant answers powered by RAG
- ✅ View Q&A history during session
- ✅ Export Q&A history as `.txt` or `.csv`

---

## 📁 Folder Structure

your_project/
├── rag_app.py # The main Streamlit app
├── docs/ # Place uploaded documents here (auto-created)
├── rag_data/ # Chroma DB folder (auto-created)
├── README.md # Setup instructions


---

## ⚙️ Setup Instructions

### 1. 🔁 Clone the Project (if applicable)

```bash
git clone https://github.com/your-repo/rag-app.git
cd rag-app
```
Or place rag_app.py and this README in any working directory.


2. 🐍 Create & Activate a Virtual Environment (optional but recommended)

# Windows

python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. 📦 Install Required Packages

```bash
pip install streamlit langchain langchain-community langchain-ollama chromadb python-docx unstructured pymupdf pandas
```

4. 🧠 Make Sure Ollama is Installed and Running
If you haven’t installed Ollama yet:

```bash
curl -fsSL https://ollama.com/install.sh | sh

```

Start the Ollama server:
```bash
ollama serve
```

Pull at least one model:
```bash
ollama pull mistral
ollama pull llama3
ollama pull phi3
```

5. ▶️ Run the App
```bash
streamlit run rag.py
```

To stop model
```bash
ollama stop mistral
ollama stop llama3
ollama stop phi3
```

Your browser will open at http://localhost:8501.

📝 How to Use
Upload Documents from the sidebar (.pdf, .txt, .md, .docx)

Select use case to control chunk size and overlap

Choose a model from Ollama (e.g., mistral, llama3, etc.)

Ask questions in the text box

View Q&A history

Download history as .txt or .csv

📂 Supported File Types
Type	Extension	Notes
PDF	.pdf	Extracted via PyMuPDF
Text	.txt	Plain UTF-8 text
Markdown	.md	Uses unstructured loader
Word Docx	.docx	Uses python-docx loader

❓ Troubleshooting
Make sure Ollama is running (ollama serve)

Pull the required models: ollama pull mistral

If documents aren't loading: ensure the docs/ folder exists or is writeable

To force reload embeddings, use the “Force Rebuild” button in the sidebar

🧩 Optional Ideas for Expansion
Add PDF export

Add persistent chat memory

Save Q&A to a database

Deploy to Streamlit Cloud (with non-Ollama LLMs like OpenAI)

