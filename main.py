import streamlit as st
import os
import shutil
import uuid # Unique IDs generate karne ke liye
from dotenv import load_dotenv
# Ab hum class ko import karenge, object ko nahi
from llm_rag.core import RAGSystem, BASE_DB_PATH 
import gc
import time


# Load environment variables
load_dotenv()

# --- Setup Directories ---
PDF_DIR = "data/uploaded_docs"
os.makedirs(PDF_DIR, exist_ok=True) 
os.makedirs(BASE_DB_PATH, exist_ok=True) # New: Saare vector stores ko store karne ke liye

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Session-Based RAG Chatbot", layout="wide")
st.title("📂 Session-Based RAG Chatbot")
st.caption("Each chat session is isolated to its own set of uploaded documents.")

# --- Session State Management ---

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4()) # Unique ID for the current session
    st.session_state.rag_system = RAGSystem(st.session_state.session_id)
    st.session_state.messages = []
    st.session_state.uploaded_file_name = None

def start_new_session():
    """Resets the state to start a new isolated chat."""
    # Purana data vector store mein save rahega, bas hum naya session shuru kar rahe hain
    st.session_state.session_id = str(uuid.uuid4())
    # FIX: st.session_id ki jagah st.session_state.session_id use kiya
    st.session_state.rag_system = RAGSystem(st.session_state.session_id) # Naye ID ke saath naya object
    st.session_state.messages = []
    st.session_state.uploaded_file_name = None
    st.success("🎉 New Chat Session started! Upload a document.")

def clear_current_session_data():
    """Deletes the vector store and data for the currently active session."""
    if st.session_state.rag_system:
        # **IMP FIX**: ChromaDB reference ko memory se hatao
        # Taki OS files ko release kar de.
        st.session_state.rag_system.vector_store = None
        
        # Thoda sa time do OS ko file lock release karne ke liye (optional but safer)
        import time
        time.sleep(0.5) 
        
        session_db_path = st.session_state.rag_system.db_path
        if os.path.exists(session_db_path):
            # Ab shutil.rmtree chalao
            shutil.rmtree(session_db_path)
            st.warning(f"Data for session '{st.session_state.session_id}' deleted successfully.")
    
    # Naya session shuru karo (jismein RAGSystem ka naya object banega)
    start_new_session()


def process_and_ingest(uploaded_file):
    """Saves PDF and ingests it into the current RAG session."""
    try:
        # File ko sirf ek baar save karo, har session ke liye nahi
        file_path = os.path.join(PDF_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.info(f"File '{uploaded_file.name}' saved. Starting ingestion into session...")

        # Ingest document into the current session's vector store
        st.session_state.rag_system.ingest_document(file_path)
        st.session_state.uploaded_file_name = uploaded_file.name
        
        st.success("✅ Knowledge Base Ready for this session! Ask your questions.")
    
    except Exception as e:
        st.error(f"Error during ingestion: {e}")

def handle_user_query(text_query):
    # ... (Query handling logic same rahega) ...
    if not text_query:
        return
    
    st.session_state.messages.append({"role": "user", "content": text_query})
    
    with st.chat_message("user"):
        st.markdown(text_query)
    
    # Get answer from RAG
    with st.spinner("Thinking..."):
        ai_response_text = st.session_state.rag_system.query(text_query)
    
    st.session_state.messages.append({"role": "assistant", "content": ai_response_text})
    
    with st.chat_message("assistant"):
        st.markdown(ai_response_text)


# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("1. Session Control")
    st.button("➕ Start New Chat", on_click=start_new_session)
    st.button("❌ Clear Current Session Data", on_click=clear_current_session_data)
    
    st.markdown("---")
    st.header("2. Knowledge Upload")
    
    if st.session_state.uploaded_file_name:
        st.info(f"Loaded Doc: {st.session_state.uploaded_file_name}")
    else:
        uploaded_file = st.file_uploader(
            "Upload a PDF for the current session:",
            type=['pdf'],
            key="file_uploader"
        )
        if uploaded_file:
            process_and_ingest(uploaded_file)
    
    st.markdown(f"**Current Session ID:** {st.session_state.session_id[:8]}...")


# --- Main Chat Interface ---

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat Input
if st.session_state.uploaded_file_name:
    text_prompt = st.chat_input("Ask a question about the document...", key="text_input")
    if text_prompt:
        handle_user_query(text_prompt)
else:
    st.warning("Upload a document in the sidebar to start chatting!")

# --- Running the App ---
# Terminal mein yeh command run karein:
# streamlit run app.py
