import streamlit as st

def yolo_classify(image, model):
    results = model.predict(source=image, save=False)[0]
    class_id = int(results.probs.top1)
    class_name = results.names[class_id]
    prob = float(results.probs.top1conf)
    return class_name, prob

def classify_quality(image, model):
    label, prob = yolo_classify(image, model)
    st.info(f"Classified as: **{label}** ({prob:.2%})")
    return label

def detect_rotation(image, model):
    label, prob = yolo_classify(image, model)
    st.info(f"Detected rotation: **{label}** ({prob:.2%})")
    if label == "90":
        return image.rotate(90, expand=True)
    elif label == "180":
        return image.rotate(180, expand=True)
    elif label == "270":
        return image.rotate(-90, expand=True)
    return image
