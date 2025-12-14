from typing import List
from functools import lru_cache
from PIL import Image
from sentence_transformers import SentenceTransformer, util

class CLIPModelWrapper:
    
    def __init__(self):
        # Must match the text embedding model name exactly
        self.model_name = "clip-ViT-L-14"
        self.device = "cpu"
            
        print(f"Loading CLIP model {self.model_name} on {self.device}...")
        
        try:
            # sentence-transformers handles the CLIP model loading
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise e

    def get_image_embedding(self, image_path: str) -> List[float]:
        """
        Generate normalized embedding for an image.
        """
        try:
            image = Image.open(image_path)
            # SentenceTransformer encodes images directly if passed to encode
            embedding = self.model.encode(image, convert_to_tensor=False, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Error embedding image {image_path}: {e}")
            return []

    def get_image_label(self, image_path: str, candidates: List[str]) -> str:
        """
        Zero-shot classification of image against candidate labels.
        """
        try:
            image = Image.open(image_path)
            
            # Encode image
            img_emb = self.model.encode(image, convert_to_tensor=True, normalize_embeddings=True)
            
            # Encode text candidates
            text_embs = self.model.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
            
            # Compute cosine similarities
            cos_scores = util.cos_sim(img_emb, text_embs)[0]
            
            # Find best match
            best_score_idx = cos_scores.argmax()
            return candidates[best_score_idx]
        
        except Exception as e:
            print(f"Error classifying image {image_path}: {e}")
            return "unknown"


@lru_cache()
def get_clip_model() -> CLIPModelWrapper:
    return CLIPModelWrapper()
