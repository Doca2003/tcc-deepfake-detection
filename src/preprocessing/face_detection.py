from facenet_pytorch import MTCNN
from PIL import Image
import os

mtcnn = MTCNN(image_size=224, margin=20, post_process=False)

def detect_and_crop_face(image_path, output_path):
    img = Image.open(image_path).convert("RGB")
    face=mtcnn(img, save_path=output_path)
    return face is not None
