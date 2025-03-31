from utils.image_utils import load_images_in_batch, preprocess_image
from models.clip import CLIPModel
from pathlib import Path
import torch

def main():
    print("Loading Model...")
    clip_model = CLIPModel()

    print("Loading Images...")
    images = load_images_in_batch()
    print(len(images), images[0])

    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Calculating Embeddings...")
    for img_name, image in images:
        print(f"Processing {img_name}...")
        preprocessed_image = preprocess_image(image, clip_model)
        image_embedding = clip_model.get_image_embedding(preprocessed_image)

        torch.save(image_embedding, output_dir / f'{img_name}_embedding.pt')
        print(f"Saved embedding for {img_name}")

    print("All embeddings saved.")



if __name__ == "__main__":
    main()