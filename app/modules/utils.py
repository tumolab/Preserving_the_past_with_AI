import base64
from collections import Counter
import zipfile
import io

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

def count_classes(class_names):
    return Counter(class_names)

def create_zip(crops):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
        for name, crop in crops:
            img_buffer = io.BytesIO()
            crop.save(img_buffer, format="JPEG")
            zf.writestr(name, img_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer