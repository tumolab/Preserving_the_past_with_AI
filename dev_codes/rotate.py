import os
import random
from PIL import Image

def rotate_and_sort_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    angles = [0, 90, 180, 270]

    for angle in angles:
        os.makedirs(os.path.join(output_folder, str(angle)), exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            angle = random.choice(angles)
            rotated_img = img.rotate(-angle, expand=True)
            save_path = os.path.join(output_folder, str(angle), filename)
            rotated_img.save(save_path)

rotate_and_sort_images("/path/to/input", "/path/to/output")
