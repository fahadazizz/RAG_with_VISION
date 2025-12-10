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
from agents.document_agent import DocumentAgent, get_document_agent
from chains.rag_chain import RAGChain, get_rag_chain


# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy initialization of components
_document_agent = None
_rag_chain = None


def get_agent():
    """Get or create document agent."""
    global _document_agent
    if _document_agent is None:
        _document_agent = get_document_agent()
    return _document_agent


def get_chain():
    """Get or create RAG chain."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = get_rag_chain()
    return _rag_chain


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for querying."""
    question: str = Field(..., description="Question to ask")
    filter: Optional[dict] = Field(None, description="Metadata filter")


class QueryResponse(BaseModel):
    """Response model for queries."""
    answer: str
    sources: list
    query: str
    timestamp: str


class URLIngestRequest(BaseModel):
    """Request model for URL ingestion."""
    url: str = Field(..., description="URL to ingest")


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    status: str
    source: str
    chunks_created: int
    timestamp: str


@app.post("/upload", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(..., description="PDF or DOCX file to upload"),
):
    """
    Upload and ingest a document (PDF or DOCX).
    
    The document will be processed, chunked, embedded, and stored in the vector database.
    """
    # Validate file type
    allowed_extensions = {".pdf", ".docx"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}",
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Ingest the document
        agent = get_agent()
        result = agent.ingest_file(tmp_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return IngestResponse(
            status=result["status"],
            source=file.filename,
            chunks_created=result["chunks_created"],
            timestamp=result["timestamp"],
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}",
        )


@app.post("/upload-url", response_model=IngestResponse)
async def upload_url(request: URLIngestRequest):
    """
    Ingest content from a URL.
    
    The webpage content will be extracted, chunked, embedded, and stored.
    """
    try:
        agent = get_agent()
        result = agent.ingest_url(request.url)
        
        return IngestResponse(
            status=result["status"],
            source=request.url,
            chunks_created=result["chunks_created"],
            timestamp=result["timestamp"],
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing URL: {str(e)}",
        )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    Retrieves relevant documents and generates an answer using the LLM.
    """
    try:
        chain = get_chain()
        response = chain.query(
            question=request.question,
            filter=request.filter,
        )
        
        return QueryResponse(
            answer=response.answer,
            sources=response.sources,
            query=response.query,
            timestamp=datetime.now().isoformat(),
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete all chunks associated with a filename.
    """
    try:
        agent = get_agent()
        result = agent.delete_file(filename)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}",
        )


# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
