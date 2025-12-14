import os
import tempfile
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

from config import get_settings
from agents.document_agent import get_document_agent
from chains.rag_chain import get_rag_chain


app = FastAPI(title="VRAG - Vision RAG API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_document_agent = None
_rag_chain = None


def get_agent():
    global _document_agent
    if _document_agent is None:
        _document_agent = get_document_agent()
    return _document_agent


def get_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = get_rag_chain()
    return _rag_chain


class ChatResponse(BaseModel):
    answer: str
    sources: list
    query: Optional[str] = None
    timestamp: str


class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks_created: int
    images_indexed: int
    timestamp: str


class ImageIngestResponse(BaseModel):
    status: str
    filename: str
    label: str
    timestamp: str


@app.post("/upload", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(..., description="PDF or DOCX file"),
):
    """Upload and ingest a document (PDF or DOCX) with images."""
    allowed_extensions = {".pdf", ".docx"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}",
        )
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        agent = get_agent()
        result = agent.ingest_file(tmp_path, original_filename=file.filename)
        
        os.unlink(tmp_path)
        
        return IngestResponse(
            status=result["status"],
            filename=file.filename,
            chunks_created=result["chunks_created"],
            images_indexed=result.get("images_indexed", 0),
            timestamp=result["timestamp"],
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/upload-image", response_model=ImageIngestResponse)
async def upload_image(
    file: UploadFile = File(..., description="Image file"),
):
    """Upload and ingest a standalone image using CLIP."""
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported: {file_ext}. Allowed: {allowed_extensions}",
        )
    
    try:
        from pathlib import Path
        from langchain_core.documents import Document
        from models.clip_model import get_clip_model
        from utils.vector_store import get_vector_store
        
        img_dir = Path("static/images")
        img_dir.mkdir(parents=True, exist_ok=True)
        
        image_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        image_path = img_dir / image_filename
        
        content = await file.read()
        with open(image_path, "wb") as f:
            f.write(content)
        
        clip_model = get_clip_model()
        candidates = ["chart", "diagram", "table", "screenshot", "photograph", "document page", "plot", "graph", "infographic"]
        label = clip_model.get_image_label(str(image_path), candidates)
        embedding = clip_model.get_image_embedding(str(image_path))
        
        if not embedding:
            raise HTTPException(status_code=500, detail="Failed to embed image")
        
        timestamp = datetime.now().isoformat()
        img_meta = {
            "source": file.filename,
            "filename": file.filename,
            "type": "image",
            "label": label,
            "timestamp": timestamp
        }
        
        img_text = f"Image: {label} from {file.filename}"
        img_doc = Document(page_content=img_text, metadata=img_meta)
        
        vector_store = get_vector_store()
        vector_store.add_image_documents([img_doc], [embedding])
        
        # Delete image after embedding
        try:
            os.unlink(str(image_path))
        except:
            pass
        
        return ImageIngestResponse(
            status="success",
            filename=file.filename,
            label=label,
            timestamp=timestamp
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(
    question: Optional[str] = Form(None, description="Your question"),
    image: Optional[UploadFile] = File(None, description="Optional image for visual search"),
):
    """Multimodal chat - supports text, image, or text+image queries."""
    try:
        chain = get_chain()
        
        image_path = None
        if image:
            file_ext = os.path.splitext(image.filename)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                content = await image.read()
                tmp_file.write(content)
                image_path = tmp_file.name
        
        response = chain.query(question=question, image_query_path=image_path)
        
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)
        
        return ChatResponse(
            answer=response.answer,
            sources=response.sources,
            query=response.query,
            timestamp=datetime.now().isoformat(),
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("api:app", host=settings.api_host, port=settings.api_port, reload=settings.api_reload)
