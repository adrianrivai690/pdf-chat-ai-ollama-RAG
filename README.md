# 🧠 Local PDF RAG Chatbot (Ollama + Phi-3 + Chroma)

This project is a lightweight Retrieval-Augmented Generation (RAG) chatbot that runs **fully locally**.  
It uses:

- **Ollama** — for local embedding + LLM inference
- **Phi-3 3.8B** — chat model
- **nomic-embed-text** — embedding model
- **ChromaDB** — persistent vector database
- **Streamlit UI** — simple chat frontend

You upload a PDF → the script splits it into chunks → embeds → stores in Chroma → you can ask questions over it.

---

## 📦 Features

✔ Local, private RAG  
✔ Uses small, fast local models  
✔ Persistent embeddings  
✔ Out-of-scope detection  
✔ Conversation memory  
✔ Simple UI

---

## 🛠️ Requirements

- Python **3.10+**
- **Ollama installed** (important!)
- Models pulled locally:
```
ollama pull phi3:3.8b
ollama pull nomic-embed-text
```
## Install Dependencies
```
pip install -r requirements.txt
```
## 🚀 How to Use

### 1. Ingest a PDF

This embeds the document and stores chunks in Chroma:

```powershell
python -m app.ingest data/yourfile.pdf
```
#### This will:

- Load PDF
- Split into chunks
- Compute embeddings using nomic-embed-text
- Store them in embeddings/chroma_db

### 2.Run the Chat UI
```
streamlit run ui/streamlit_app.py
```
##