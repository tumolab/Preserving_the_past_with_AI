from ultralytics import YOLO
from openai import OpenAI
from config import (
    DETECTION_MODEL_PATH,
    CLASSIFICATION_MODEL_PATH,
    ROTATION_MODEL_PATH,
    OPENAI_API_KEY
)

def load_yolo_models():
    detection_model = YOLO(DETECTION_MODEL_PATH)
    classification_model = YOLO(CLASSIFICATION_MODEL_PATH)
    rotation_model = YOLO(ROTATION_MODEL_PATH)
    return detection_model, classification_model, rotation_model

def load_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)
