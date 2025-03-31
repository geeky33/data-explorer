from PIL import Image
from pathlib import Path
import os
import torch

def load_image(image_name, data_dir = 'src/data/images'):
    project_root = Path(__file__).parent.parent.parent

    image_path = project_root / data_dir / image_name

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    return Image.open(image_path)


def load_images_in_batch(data_dir='src/data/images'):
    project_root = Path(__file__).parent.parent.parent
    folder_path = project_root / data_dir

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    images = []
    for img_file in os.listdir(folder_path):
        img_path = folder_path / img_file
        if os.path.isfile(img_path) and img_file.lower().endswith(('png', 'jpg', 'jpeg')):
            image = Image.open(img_path)
            img_name = os.path.splitext(img_file)[0]
            images.append((img_name, image))
    
    if not images:
        raise FileNotFoundError("No images found in the specified directory.")
    
    return images


def preprocess_image(image, clip_model):
    preprocess = clip_model.preprocess

    return preprocess(image).unsqueeze(0)