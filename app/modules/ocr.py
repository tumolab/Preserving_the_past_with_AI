import pytesseract
import streamlit as st

def ocr_crop(image, lang="HYE-new"):
    text = pytesseract.image_to_string(image, lang=lang, config='--dpi 300')
    return text.strip()

def display_and_ocr(crops, binarized_crops):
    texts = {}
    for (name, crop), (_, bin_crop) in zip(crops, binarized_crops):

        w, h = crop.size
        textarea_height = int(h * 0.05)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(crop, caption=name, use_container_width=True)
        with col2:
            text = ocr_crop(bin_crop)
            texts[name] = text
            st.text_area(f"OCR Text from {name}", value=text, height=textarea_height)

    return texts
