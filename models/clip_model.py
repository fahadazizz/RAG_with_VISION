from typing import List
from functools import lru_cache
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPModelWrapper:
    
    def __init__(self):
        self.model_name = "openai/clip-vit-large-patch14"
        # Determine device
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        print(f"Loading CLIP model {self.model_name} on {self.device}...")
        
        try:
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise e

    def get_image_embedding(self, image_path: str) -> List[float]:
        """
        Generate normalized embedding for an image.
        """
        try:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Normalize
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.squeeze().tolist()
        except Exception as e:
            print(f"Error embedding image {image_path}: {e}")
            return []

    def get_image_label(self, image_path: str, candidates: List[str]) -> str:
        """
        Zero-shot classification of image against candidate labels.
        """
        try:
            image = Image.open(image_path)
            inputs = self.processor(text=candidates, images=image, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1)
            
            # Get index of highest probability
            top_prob, top_lbl_idx = probs.topk(1)
            return candidates[top_lbl_idx.item()]
        
        except Exception as e:
            print(f"Error classifying image {image_path}: {e}")
            return "unknown"


@lru_cache()
def get_clip_model() -> CLIPModelWrapper:
    return CLIPModelWrapper()
