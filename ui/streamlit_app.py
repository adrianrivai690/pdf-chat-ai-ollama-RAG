# ui/streamlit_app.py
import streamlit as st
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag_chat import answer_query, MAX_MEMORY_MESSAGES


st.set_page_config(page_title="ABB RAG Chatbot", layout="wide")
st.title("🤖 ABB Local Chatbot — Single PDF RAG")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ask something from the uploaded document:")


if st.button("Ask"):
    st.session_state.history.append({"role": "user", "content": user_input})
    last_msgs = st.session_state.history[-MAX_MEMORY_MESSAGES:]
    with st.spinner("Generating..."):
        answer = answer_query(user_input, last_msgs)
    st.session_state.history.append({"role": "bot", "content": answer})

st.write("### Conversation:")
for msg in st.session_state.history:
    role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
    st.write(f"**{role}:** {msg['content']}")

if st.button("Clear Chat"):
    st.session_state.history = []
    st.rerun()