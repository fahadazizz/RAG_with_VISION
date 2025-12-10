import streamlit as st
import requests
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


def upload_file(file):
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def upload_url(url: str):
    try:
        response = requests.post(
            f"{API_BASE_URL}/upload-url",
            json={"url": url}
        )
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def query_rag(question: str):
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question}
        )
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


if "messages" not in st.session_state:
    st.session_state.messages = []


st.markdown('<h1 class="main-header">🤖 RAG Chatbot</h1>', unsafe_allow_html=True)

# SIDEBAR - Chat History Only
with st.sidebar:
    st.header("💬 Chat History")
    
    st.divider()
    
    # Display chat history
    if st.session_state.messages:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content'][:50]}...")
            else:
                st.markdown(f"**AI:** {message['content'][:50]}...")
            st.caption(f"Message {i+1}")
            st.divider()
    else:
        st.caption("No chat history yet")
    
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        st.rerun()


# MAIN AREA - Document Upload, URL, and Query
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file",
        type=["pdf", "docx"],
        help="Upload a document to add to the knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("📤 Process Document"):
            with st.spinner("Processing document..."):
                result = upload_file(uploaded_file)
                
                if result.get("status") == "success":
                    st.success(f"✅ Uploaded: {result.get('filename')}")
                    st.info(f"Created {result.get('chunks_created')} chunks")
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

with col2:
    st.subheader("🌐 Ingest from URL")
    url_input = st.text_input(
        "Enter URL",
        placeholder="https://example.com/article",
        help="Enter a URL to extract and ingest content"
    )
    
    if st.button("📥 Ingest URL"):
        with st.spinner("Fetching and processing URL..."):
            result = upload_url(url_input)
            
            if result.get("status") == "success":
                st.success(f"✅ Ingested URL")
                st.info(f"Created {result.get('chunks_created')} chunks")
            else:
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

st.divider()

# Chat Interface
st.subheader("💬 Ask Questions")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source.get('filename', 'Unknown')}")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = query_rag(prompt)
            
            if "error" in result:
                response = f"❌ Error: {result['error']}"
                sources = []
            else:
                response = result.get("answer", "No response received")
                sources = result.get("sources", [])
            
            st.markdown(response)
            
            if sources:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:** {source.get('filename', 'Unknown')}")
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
    })

st.divider()
st.caption("Built with LangChain, Pinecone, and Ollama | RAG Chatbot v1.0")
