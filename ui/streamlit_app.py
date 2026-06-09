import streamlit as st
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag_chat import answer_query, MAX_MEMORY_MESSAGES


st.set_page_config(
    page_title="ABB RAG Chatbot", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 Controls")
    st.markdown("Welcome to the **ABB Local Chatbot**. This uses a local RAG pipeline to answer questions based on your documents.")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="primary"):
        st.session_state.history = []
        st.rerun()
        
    st.divider()
    st.caption("Powered by Ollama & ChromaDB")

# --- Main Interface ---
st.title("ABB Local Chatbot — Single PDF RAG")
st.markdown("Ask anything about your document context below.")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat messages from history on app rerun
for msg in st.session_state.history:
    role = "assistant" if msg["role"] == "bot" else "user"
    with st.chat_message(role):
        st.markdown(msg["content"])

# React to user input
if prompt := st.chat_input("Ask something from the uploaded document..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.history.append({"role": "user", "content": prompt})
    
    # Get recent history
    last_msgs = st.session_state.history[-MAX_MEMORY_MESSAGES:]
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_query(prompt, last_msgs)
        st.markdown(answer)
        
    # Add assistant response to chat history
    st.session_state.history.append({"role": "bot", "content": answer})