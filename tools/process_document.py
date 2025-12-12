from datetime import datetime
from langchain_core.documents import Document
from tools.load_documents import load_document
from tools.clean_text import clean_document_text
from tools.utils.text_chunker import TextChunker


def process_document(
    source: str,
    original_filename: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    clean_text: bool = True,
) -> list[Document]:
    """
    Complete document processing pipeline: load → clean → chunk.
    This is the main entry point for document ingestion.
    
    Args:
        source: File path or URL
        original_filename: Original filename (if source is a temp file)
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        clean_text: Whether to apply text cleaning
        
    Returns:
        List of Document chunks ready for embedding
    """
    documents = load_document(source)
    
    if original_filename:
        filename = original_filename
    elif source.startswith(("http://", "https://")):
        filename = source
    else:
        from pathlib import Path
        filename = Path(source).name
    
    image_infos = {}
    for doc in documents:
        if "image_paths" in doc.metadata:
            for path in doc.metadata["image_paths"]:
                if path not in image_infos:
                    image_infos[path] = {
                        "page": doc.metadata.get("page", 0),
                        "filename": doc.metadata.get("filename", filename)
                    }

    
    image_docs = []
    image_embeddings = []
    
    timestamp = datetime.now().isoformat()
    
    try:
        from models.clip_model import get_clip_model
        import os
        
        clip_model = get_clip_model()
        candidates = ["chart", "diagram", "table", "screenshot", "photograph", "document page", "plot", "graph", "infographic"]
        
        print(f"Processing {len(image_infos)} images with CLIP...")
        
        for img_path, info in image_infos.items():
            label = clip_model.get_image_label(img_path, candidates)
            embedding = clip_model.get_image_embedding(img_path)
            
            if embedding:
                from pathlib import Path
                img_name = Path(img_path).name
                page_num = info["page"]
                
                img_meta = {
                    "source": info["filename"],
                    "filename": info["filename"],
                    "type": "image",
                    "label": label,
                    "page": page_num,
                    "timestamp": timestamp
                }
                
                # Clean, simple text for vector storage
                img_text = f"Image: {label} from {info['filename']} page {page_num}"
                
                img_doc = Document(
                    page_content=img_text,
                    metadata=img_meta
                )
                image_docs.append(img_doc)
                image_embeddings.append(embedding)
            
            # Delete image after embedding
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    print(f"Deleted: {img_path}")
            except Exception as del_e:
                print(f"Could not delete {img_path}: {del_e}")
                
    except ImportError:
        print("CLIP model not found. Skipping image processing.")
    except Exception as e:
        print(f"Error processing images with CLIP: {e}")

    all_chunks = []
    for doc in documents:
        text = doc.page_content
        
        if clean_text:
            text = clean_document_text(text)
        
        if not text.strip():
            continue
        
        metadata = {
            "filename": filename,
            "timestamp": timestamp,
        }
        
        if "page" in doc.metadata:
            metadata["page"] = doc.metadata["page"]
        
        if "image_paths" in doc.metadata:
            metadata["image_paths"] = doc.metadata["image_paths"]
        
        # Chunk the text
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk(text, metadata)
        all_chunks.extend(chunks)
    
    return all_chunks, (image_docs, image_embeddings)
