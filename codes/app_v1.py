import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load models (adjust paths or use model names)
detection_model = YOLO("yolov8n.pt")       # detection model
classification_model = YOLO("yolov8n-cls.pt")  # classification model

# --- Inference functions ---
def yolo_detect(image, conf_thresh):
    results = detection_model.predict(source=image, conf=conf_thresh, save=False, imgsz=640)
    result_img = results[0].plot()
    return Image.fromarray(result_img[..., ::-1])

def yolo_classify(image):
    results = classification_model.predict(source=image, save=False)
    class_id = int(results[0].probs.top1)
    class_name = results[0].names[class_id]
    return class_name

# --- UI ---
st.sidebar.title("YOLO Demo")
task = st.sidebar.selectbox("Choose Task", ["Detection", "Classification"])
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if task == "Detection":
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

run = st.sidebar.button("Run")

st.title("YOLO Demo")
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original")

    if run:
        if task == "Detection":
            det_img = yolo_detect(image, conf_thresh)
            with col2:
                st.image(det_img, caption="Detection Output")

        elif task == "Classification":
            label = yolo_classify(image)
            with col2:
                st.write(f"Classified as: {label}")