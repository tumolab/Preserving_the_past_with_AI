import streamlit as st
from PIL import Image
from ultralytics import YOLO
import io
import zipfile
import numpy as np
import cv2
from collections import Counter

# Load models
detection_model = YOLO("PageDetectNewspaper_best.pt")  # segmentation
classification_model = YOLO("yolov8n-cls.pt")
rotation_model = YOLO("classify_rotation.pt")

# --- Functions ---
def yolo_detect(image, conf_thresh=0.7):
    results = detection_model.predict(source=image, conf=conf_thresh, save=False, imgsz=640, iou=0.3)
    result_img = results[0].plot()
    masks = results[0].masks.xy if results[0].masks else []
    return Image.fromarray(result_img[..., ::-1]), masks

def yolo_classify(image, custom_model):
    results = custom_model.predict(source=image, save=False)[0]
    class_id = int(results.probs.top1)
    class_name = results.names[class_id]
    prob = float(results.probs.top1conf)
    return class_name, prob

def order_corners(pts):
    pts = np.array(pts).reshape(-1, 2).astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(s)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def crop_objects(image, segments):
    crops = []
    img_np = np.array(image)

    for i, seg in enumerate(segments):
        if len(seg) < 4:
            continue
        try:
            box = order_corners(seg)
            width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
            height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
            dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
            M = cv2.getPerspectiveTransform(box, dst_pts)
            warped = cv2.warpPerspective(img_np, M, (width, height))
            pil_crop = Image.fromarray(warped)
            crops.append((f"object_{i+1}.jpg", pil_crop))
        except Exception as e:
            print(f"Polygon {i} failed: {e}")
            continue
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

        if label == "sports_car":
            st.warning("We have detected that the image is not in good quality.")
            st.session_state["waiting_for_user_confirm"] = True
        else:
            st.session_state["run_detection"] = True

    if st.session_state.get("waiting_for_user_confirm", False):
        if st.button("Process still", key="manual_confirm"):
            st.session_state["run_detection"] = True
            st.session_state["waiting_for_user_confirm"] = False

    if st.session_state.get("run_detection", False):

        det_img, masks = yolo_detect(selected_image)
        with col2:
            st.image(det_img, caption="Detection Output")

        crops = crop_objects(selected_image, masks)
        # if crops:
        #     zip_file = create_zip(crops)
        #     st.download_button(
        #         label="📥 Download Crops as ZIP",
        #         data=zip_file,
        #         file_name="crops.zip",
        #         mime="application/zip"
        #     )
        if crops:
            st.markdown("### Detected Crops")
            for name, crop in crops:
                st.image(crop, caption=name)


        st.session_state["run_detection"] = False
