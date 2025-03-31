import torch
import clip

class CLIPModel:
    def __init__(self, model_name = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def get_image_embedding(self, preprocessed_image):
        with torch.no_grad():
            image_features = self.model.encode_image(preprocessed_image.to(self.device))
            # Normalize the features
            image_embedding = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_embedding