import numpy as np
from models.embedding_model import get_embedding_model
from models.clip_model import get_clip_model
from PIL import Image
import requests
from io import BytesIO

def verify_alignment():
    print("Initializing models...")
    text_model = get_embedding_model()
    image_model = get_clip_model()
    
    # Test Data: "A dog"
    query_text = "a photo of a dog"
    
    # Download a dog image for testing
    img_url = "https://images.unsplash.com/photo-1543466835-00a7907e9de1?q=80&w=1000&auto=format&fit=crop"
    print(f"Downloading test image from {img_url}...")
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    img_path = "test_dog.jpg"
    img.save(img_path)
    
    # 1. Embed Text
    print("Embedding text...")
    text_emb = text_model.embed_query(query_text)
    
    # 2. Embed Image
    print("Embedding image...")
    image_emb = image_model.get_image_embedding(img_path)
    
    # 3. Compute Similarity
    print("Computing similarity...")
    # Convert to numpy and normalize (just in case)
    v1 = np.array(text_emb)
    v2 = np.array(image_emb)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        print("Error: Zero magnitude vector found.")
        return
        
    sim = np.dot(v1, v2) / (norm1 * norm2)
    
    print(f"\n--- Results ---")
    print(f"Text: '{query_text}'")
    print(f"Image: {img_path}")
    print(f"Cosine Similarity: {sim:.4f}")
    
    if sim > 0.2:
        print("SUCCESS: High similarity detected. Models are aligned.")
    else:
        print("FAILURE: Low similarity. Models might be misaligned.")

if __name__ == "__main__":
    verify_alignment()
