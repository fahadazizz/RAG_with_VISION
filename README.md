# RAG Chatbot Pipeline

A complete end-to-end Retrieval-Augmented Generation (RAG) chatbot system built with LangChain, Pinecone, and Ollama.

## 🌟 Features

- **Document Ingestion**: Upload PDF, DOCX files or ingest content from URLs
- **Smart Chunking**: Intelligent text splitting with context preservation
- **Vector Storage**: Pinecone for scalable vector storage
- **Local LLM**: Ollama for privacy-focused generation
- **REST API**: FastAPI endpoints for programmatic access
- **Web Dashboard**: Beautiful Streamlit interface

## 🏗️ Architecture

```
User Document → Load → Clean → Chunk → Embed → Pinecone
                                                    ↓
User Query → Encode → Retrieve → Rerank → Augment → LLM → Response
```

## 📋 Prerequisites

1. **Python 3.12+**
2. **Ollama** installed and running
   ```bash
   # Install Ollama from https://ollama.ai
   ollama serve
   ollama pull kimi-k2:thinking  # or your preferred model
   ```
3. **Pinecone Account** with API key and index created

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd /Volumes/DataDrive/my_AGENTS_AND_MCP/rag_wtih_CLIP
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create/update `.env` file:
```env
PINECONE_API_KEY=your_api_key_here
PINECONE_INDEX_NAME=your_index_name
```

### 3. Run the System

**Option A: API + Dashboard (Recommended)**

Terminal 1 - Start API:
```bash
source .venv/bin/activate
uvicorn api:app --reload
```

Terminal 2 - Start Dashboard:
```bash
source .venv/bin/activate
streamlit run dashboard.py
```

Then open http://localhost:8501 in your browser.

**Option B: Programmatic Usage**

```python
from agents.document_agent import get_document_agent
from chains.rag_chain import get_rag_chain

# Ingest a document
agent = get_document_agent()
result = agent.ingest_file("path/to/document.pdf")
print(f"Created {result['chunks_created']} chunks")

# Query the system
rag = get_rag_chain()
response = rag.query("What is this document about?")
print(response.answer)
```

## 📁 Project Structure

```
rag_wtih_CLIP/
├── config.py              # Configuration management
├── requirements.txt       # Dependencies
├── api.py                 # FastAPI endpoints
├── dashboard.py           # Streamlit UI
├── test_pipeline.py       # Component tests
├── agents/
│   └── document_agent.py  # Document ingestion orchestrator
├── chains/
│   ├── retriever.py       # Retrieval with reranking
│   └── rag_chain.py       # Complete RAG chain
├── models/
│   ├── embedding_model.py # Sentence Transformers
│   ├── vector_store.py    # Pinecone manager
│   └── llm.py             # Ollama wrapper
├── prompts/
│   └── rag_prompts.py     # Engineered prompts
└── tools/
    └── utils/
        ├── text_cleaner.py     # Text preprocessing
        ├── text_chunker.py     # Smart chunking
        └── document_loaders.py # PDF/DOCX/URL loaders
```

## ⚙️ Configuration

Edit `config.py` or set environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding_model_name` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `llm_model_name` | `kimi-k2:thinking` | Ollama model |
| `chunk_size` | `1000` | Characters per chunk |
| `chunk_overlap` | `200` | Overlap between chunks |
| `rag_top_k` | `5` | Initial retrieval count |
| `rag_score_threshold` | `0.5` | Minimum similarity score |
| `rag_rerank_top_k` | `3` | Final context documents |

## 🔌 API Endpoints

### Upload Document
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

### Ingest URL
```bash
curl -X POST "http://localhost:8000/upload-url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

### Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'
```

## 🧪 Testing

Run the test suite to verify all components:

```bash
source .venv/bin/activate
python test_pipeline.py
```

## 🎯 Key Features Explained

### Smart Text Processing
- **Cleaning**: Removes URLs, emails, normalizes whitespace
- **Chunking**: Recursive splitting with configurable overlap
- **Context**: Optional context windows from neighboring chunks

### Intelligent Retrieval
- **Vector Search**: Semantic similarity via Pinecone
- **Score Filtering**: Removes low-relevance results
- **Reranking**: Selects top-k most relevant chunks

### Anti-Hallucination Prompts
- Strict source-based answering
- Explicit acknowledgment of limitations
- Source citation requirements

## 🛠️ Troubleshooting

**Ollama Connection Error**
```bash
# Make sure Ollama is running
ollama serve

# Check if model is available
ollama list
```

**Pinecone Connection Error**
- Verify API key in `.env`
- Check index name matches your Pinecone dashboard
- Ensure index dimension matches embedding model (384 for all-MiniLM-L6-v2)

**Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📚 Tech Stack

- **LangChain**: Orchestration framework
- **Pinecone**: Vector database
- **Ollama**: Local LLM inference
- **Sentence Transformers**: Embeddings
- **FastAPI**: REST API
- **Streamlit**: Web dashboard
- **Pydantic**: Configuration management

## 🤝 Contributing

This is a complete RAG implementation following best practices:
- Modular architecture
- Type hints throughout
- Comprehensive error handling
- Engineered prompts for quality
- Configurable components

## 📄 License

MIT License - feel free to use and modify!

## 🎓 Learn More

- [LangChain Documentation](https://python.langchain.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Ollama Documentation](https://ollama.ai/docs)
