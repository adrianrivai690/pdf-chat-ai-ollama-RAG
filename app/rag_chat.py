import os
import ollama
from chromadb import PersistentClient
from typing import List, Dict

# -----------------------
# CONFIG
# -----------------------
PERSIST_DIR = "embeddings/chroma_db"
COLLECTION_NAME = "abb_docs"
CHAT_MODEL = "phi3:3.8b"
EMBED_MODEL = "nomic-embed-text"

TOP_K = 4
SIM_THRESHOLD = 0.45

# Number of previous user + bot messages to feed the LLM
MAX_MEMORY_MESSAGES = 5


# -----------------------
# EMBEDDING
# -----------------------
def embed_query(q: str):
    return ollama.embeddings(model=EMBED_MODEL, prompt=q)["embedding"]


# -----------------------
# COLLECTION
# -----------------------
def get_collection():
    client = PersistentClient(path=PERSIST_DIR)
    return client.get_collection(COLLECTION_NAME)


# -----------------------
# VECTOR SEARCH
# -----------------------
def retrieve(query: str):
    col = get_collection()
    q_emb = embed_query(query)

    res = col.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    docs = []
    for d, m, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0]
    ):
        docs.append({
            "text": d,
            "meta": m,
            "distance": dist
        })
    return docs


# -----------------------
# OUT OF SCOPE CHECK
# -----------------------
def is_out_of_scope(results):
    if not results:
        return True
    return results[0]["distance"] > SIM_THRESHOLD


# -----------------------
# CHAT LOGIC
# -----------------------
def answer_query(query: str, history: List[Dict]):
    docs = retrieve(query)

    # If no context is similar enough
    if is_out_of_scope(docs):
        return "Out of scope. I can only answer using the uploaded document."

    ### Build context
    context = "\n\n".join(
        f"[chunk {d['meta']['chunk_idx']}] {d['text']}"
        for d in docs
    )

    ### Memory (last 5 messages)
    memory = ""
    for msg in history[-MAX_MEMORY_MESSAGES:]:
        role = "USER" if msg["role"] == "user" else "BOT"
        memory += f"{role}: {msg['content']}\n"

    ### Final prompt sent to Ollama
    prompt = f"""
SYSTEM:
You are a technical assistant.
ONLY answer using the provided document context.
If it is not in the document, return: "Out of scope."

DOCUMENT CONTEXT:
{context}

CHAT HISTORY:
{memory}

USER QUESTION:
{query}

ANSWER:
"""

    resp = ollama.generate(
        model=CHAT_MODEL,
        prompt=prompt
    )
    return resp["response"]
