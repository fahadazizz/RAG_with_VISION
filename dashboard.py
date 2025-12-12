import streamlit as st
import requests
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="VRAG - Vision RAG",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    .query-mode {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def upload_document(file):
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        return response.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def upload_image(file):
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/upload-image", files=files)
        return response.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def chat(question: str = None, image_file=None):
    """Multimodal chat - text, image, or both."""
    try:
        data = {}
        files = {}
        
        if question:
            data["question"] = question
        
        if image_file:
            files["image"] = (image_file.name, image_file.getvalue(), image_file.type)
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            data=data if data else None,
            files=files if files else None
        )
        return response.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


if "messages" not in st.session_state:
    st.session_state.messages = []


st.markdown('<h1 class="main-header">🔮 VRAG - Vision RAG</h1>', unsafe_allow_html=True)
st.caption("Multimodal RAG: Text | Image | Text + Image")

# Sidebar - Chat History
with st.sidebar:
    st.header("💬 Chat History")
    st.divider()
    
    if st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):
            role = "You" if msg["role"] == "user" else "AI"
            mode = msg.get("mode", "")
            st.markdown(f"**{role}** ({mode}): {msg['content'][:30]}...")
            st.divider()
    else:
        st.caption("No chat history yet")
    
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        st.rerun()


# Main Area - Upload Section
st.subheader("📥 Ingest Data")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**📄 Document (PDF/DOCX)**")
    doc_file = st.file_uploader("Upload document", type=["pdf", "docx"], key="doc_upload", label_visibility="collapsed")
    
    if doc_file and st.button("Ingest Document", key="btn_doc"):
        with st.spinner("Processing..."):
            result = upload_document(doc_file)
            if result.get("status") == "success":
                st.success(f"✅ {result.get('chunks_created')} chunks | {result.get('images_indexed', 0)} images")
            else:
                st.error(f"❌ {result.get('detail', 'Error')}")

with col2:
    st.markdown("**🖼️ Standalone Image**")
    img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "gif", "webp"], key="img_upload", label_visibility="collapsed")
    
    if img_file:
        st.image(img_file, width=100)
        if st.button("Ingest Image", key="btn_img"):
            with st.spinner("CLIP processing..."):
                result = upload_image(img_file)
                if result.get("status") == "success":
                    st.success(f"✅ Type: {result.get('label')}")
                else:
                    st.error(f"❌ {result.get('detail', 'Error')}")


st.divider()

# Chat Section with Query Mode Selection
st.subheader("💬 Multimodal Chat")

# Query Mode Selection
query_mode = st.radio(
    "Query Mode",
    ["📝 Text Only", "🖼️ Image Only", "📝+🖼️ Text + Image"],
    horizontal=True,
    key="query_mode"
)

# Input areas based on mode
text_input = None
query_image = None

if query_mode == "📝 Text Only":
    text_input = st.text_input("Enter your question:", key="text_query", placeholder="Ask about your documents...")

elif query_mode == "🖼️ Image Only":
    query_image = st.file_uploader(
        "Upload query image:",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        key="image_query"
    )
    if query_image:
        st.image(query_image, width=200)

else:  # Text + Image
    text_input = st.text_input("Enter your question:", key="multimodal_text", placeholder="Ask about the image...")
    query_image = st.file_uploader(
        "Upload query image:",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        key="multimodal_image"
    )
    if query_image:
        st.image(query_image, width=200)

# Submit button
if st.button("🔮 Search & Generate", type="primary"):
    if not text_input and not query_image:
        st.warning("Please provide text, image, or both.")
    else:
        # Determine mode label
        if text_input and query_image:
            mode_label = "Text+Image"
        elif query_image:
            mode_label = "Image"
        else:
            mode_label = "Text"
        
        # Add user message
        display_content = text_input if text_input else "[Image Query]"
        st.session_state.messages.append({
            "role": "user",
            "content": display_content,
            "mode": mode_label
        })
        
        with st.spinner("🔮 Processing..."):
            result = chat(text_input, query_image)
            
            if result.get("detail"):
                response = f"❌ {result['detail']}"
                sources = []
            else:
                response = result.get("answer", "No response")
                sources = result.get("sources", [])
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources,
                "mode": mode_label
            })
        
        st.rerun()

# Display chat messages
st.divider()
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.caption(f"Mode: {msg.get('mode', 'Unknown')}")
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    icon = "🖼️" if src.get("type") == "image" else "📄"
                    st.markdown(f"{icon} **{i}.** {src.get('filename', 'Unknown')}")


st.divider()
st.caption("VRAG v2.0 | LangChain + CLIP + Pinecone + Ollama")
