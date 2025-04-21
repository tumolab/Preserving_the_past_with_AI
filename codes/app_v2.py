import streamlit as st
from PIL import Image
from ultralytics import YOLO
import io
import zipfile
from collections import Counter
import pandas as pd

# Load models
detection_model = YOLO("PageDetectNewspaper_best.pt")
classification_model = YOLO("yolov8n-cls.pt")
rotation_model = YOLO("classify_rotation.pt")

# --- Functions ---
def yolo_detect(image, conf_thresh=0.7):
    results = detection_model.predict(source=image, conf=conf_thresh, save=False, imgsz=640, iou=0.3)
    result_img = results[0].plot()
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    names = results[0].names
    class_names = [names[i] for i in class_ids]
    return Image.fromarray(result_img[..., ::-1]), boxes, class_names

def yolo_classify(image, custom_model):
    results = custom_model.predict(source=image, save=False)
    class_id = int(results[0].probs.top1)
    class_name = results[0].names[class_id]
    prob = float(results[0].probs.top1conf)
    return class_name, prob

def crop_objects(image, boxes):
    crops = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop = image.crop((x1, y1, x2, y2))
        crops.append((f"object_{i+1}.jpg", crop))
    return crops

def create_zip(crops):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
        for name, crop in crops:
            img_buffer = io.BytesIO()
            crop.save(img_buffer, format="JPEG")
            zf.writestr(name, img_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

def count_classes(class_names):
    return Counter(class_names)

# --- UI ---
st.sidebar.title("Menu")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

st.title("Image Processing Pipeline")
st.markdown("### Upload -> Classify -> Detection -> Export")

if uploaded_file:
    selected_image = Image.open(uploaded_file).convert("RGB")
    selected_name = uploaded_file.name

    col1, col2 = st.columns(2)
    with col1:
        st.image(selected_image, caption=f"Selected: {selected_name}")

    if st.button("Run", key="run_btn"):
        label, prob = yolo_classify(selected_image, classification_model)
        st.session_state["classification_label"] = label
        st.session_state["classification_done"] = True

        st.info(f"Classified as: **{label}** ({prob:.2%})")

        # ---- Rotation Classification Begins ----
        orientation_label, orientation_prob = yolo_classify(selected_image, rotation_model)
        st.info(f"Detected rotation: **{orientation_label}** ({orientation_prob:.2%})")

        if orientation_label == "90":
            st.warning("Rotating image -90 degrees")
            selected_image = selected_image.rotate(90, expand=True)
        elif orientation_label == "180":
            st.warning("Rotating image 180 degrees")
            selected_image = selected_image.rotate(180, expand=True)
        elif orientation_label == "270":
            st.warning("Rotating image 90 degrees")
            selected_image = selected_image.rotate(-90, expand=True)
        # ---- Rotation Classification Ends ----

        if label == "sports_car":  # Temporary class, waiting for real classifier
            st.warning("We have detected that the image is not in good quality.")
            st.session_state["waiting_for_user_confirm"] = True
        else:
            st.session_state["run_detection"] = True

    if st.session_state.get("waiting_for_user_confirm", False):
        if st.button("Process still", key="manual_confirm"):
            st.session_state["run_detection"] = True
            st.session_state["waiting_for_user_confirm"] = False

    if st.session_state.get("run_detection", False):

        det_img, boxes, class_names = yolo_detect(selected_image)
        with col2:
            st.image(det_img, caption="Detection Output")

        crops = crop_objects(selected_image, boxes)
        if crops:
            zip_file = create_zip(crops)
            st.download_button(
                label="📥 Download Crops as ZIP",
                data=zip_file,
                file_name="crops.zip",
                mime="application/zip"
            )

        st.session_state["run_detection"] = False
