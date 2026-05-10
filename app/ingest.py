import os
import uuid
from chromadb import PersistentClient  # Perbaikan 1: Menambahkan nama library chromadb
import ollama                          # Perbaikan 2: Mengimpor library ollama yang benar

from app.utils import load_pdf_as_documents, split_documents

PERSIST_DIR = "embeddings/chroma_db"
COLLECTION_NAME = "abb_docs"
EMBED_MODEL = "nomic-embed-text"


def embed_chunks(texts):
    embeddings = []
    for t in texts:
        emb = ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"]
        embeddings.append(emb)
    return embeddings


def build_chroma_from_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"📄 Loading PDF: {pdf_path}")
    docs = load_pdf_as_documents(pdf_path)
    chunks = split_documents(docs)
    print(f"✔ Created {len(chunks)} chunks")

    print("📦 Starting Chroma...")
    client = PersistentClient(path=PERSIST_DIR)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except:
        collection = client.create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # IMPORTANT FOR SIMILARITY SEARCH
        )

    texts = [c.page_content for c in chunks]
    ids = [str(uuid.uuid4()) for _ in texts]
    metas = [{"source": chunks[i].metadata["source"], "chunk_idx": i} for i in range(len(chunks))]

    print("🧠 Embedding chunks (Ollama local model)...")
    embeddings = embed_chunks(texts)

    print("💾 Adding to Chroma...")
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metas,
        embeddings=embeddings
    )

    print("🎉 DONE — Local Chroma DB built successfully!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m app.ingest data/<file>.pdf")
        exit()

    build_chroma_from_pdf(sys.argv[1])