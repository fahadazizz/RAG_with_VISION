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


def chat(question: str, image_file=None):
    try:
        data = {"question": question}
        files = {}
        
        if image_file:
            files["image"] = (image_file.name, image_file.getvalue(), image_file.type)
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            data=data,
            files=files if files else None
        )
        return response.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


if "messages" not in st.session_state:
    st.session_state.messages = []


st.markdown('<h1 class="main-header">🔮 VRAG - Vision RAG</h1>', unsafe_allow_html=True)
st.caption("Multimodal RAG with Text + Image Understanding")

# Sidebar - Chat History
with st.sidebar:
    st.header("💬 Chat History")
    st.divider()
    
    if st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):
            role = "You" if msg["role"] == "user" else "AI"
            icon = "🖼️ " if msg.get("has_image") else ""
            st.markdown(f"**{role}:** {icon}{msg['content'][:40]}...")
            st.divider()
    else:
        st.caption("No chat history yet")
    
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        st.rerun()


# Main Area - Upload Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Document")
    doc_file = st.file_uploader(
        "PDF or DOCX",
        type=["pdf", "docx"],
        key="doc_upload"
    )
    
    if doc_file and st.button("📤 Ingest Document", key="btn_doc"):
        with st.spinner("Processing..."):
            result = upload_document(doc_file)
            if result.get("status") == "success":
                st.success(f"✅ {result.get('filename')}")
                st.info(f"📝 {result.get('chunks_created')} chunks | 🖼️ {result.get('images_indexed', 0)} images")
            else:
                st.error(f"❌ {result.get('detail', 'Error')}")

with col2:
    st.subheader("🖼️ Upload Image")
    img_file = st.file_uploader(
        "Image file",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        key="img_upload"
    )
    
    if img_file:
        st.image(img_file, width=150)
        if st.button("📤 Ingest Image", key="btn_img"):
            with st.spinner("Processing with CLIP..."):
                result = upload_image(img_file)
                if result.get("status") == "success":
                    st.success(f"✅ Type: {result.get('label')}")
                else:
                    st.error(f"❌ {result.get('detail', 'Error')}")


st.divider()

# Chat Section
st.subheader("💬 Chat")

# Optional image for query
query_image = st.file_uploader(
    "🖼️ Attach image (optional)",
    type=["jpg", "jpeg", "png", "gif", "webp"],
    key="query_img",
    help="Attach an image for multimodal search"
)

if query_image:
    st.image(query_image, width=150, caption="Query Image")

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("has_image"):
            st.caption("🖼️ Image attached")
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    icon = "🖼️" if src.get("type") == "image" else "📄"
                    st.markdown(f"{icon} **{i}.** {src.get('filename', 'Unknown')}")

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    has_image = query_image is not None
    
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "has_image": has_image
    })
    
    with st.chat_message("user"):
        if has_image:
            st.caption("🖼️ Image attached")
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔮 Thinking..."):
            result = chat(prompt, query_image)
            
            if result.get("detail"):
                response = f"❌ {result['detail']}"
                sources = []
            else:
                response = result.get("answer", "No response")
                sources = result.get("sources", [])
            
            st.markdown(response)
            
            if sources:
                with st.expander("📚 Sources"):
                    for i, src in enumerate(sources, 1):
                        icon = "🖼️" if src.get("type") == "image" else "📄"
                        st.markdown(f"{icon} **{i}.** {src.get('filename', 'Unknown')}")
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
    })


st.divider()
st.caption("VRAG v2.0 | LangChain + CLIP + Pinecone + Ollama")
