import streamlit as st
from PIL import Image

from modules.load_models import load_yolo_models, load_openai_client
from modules.detection import yolo_detect, crop_objects
from modules.classification import classify_quality, detect_rotation
from modules.ocr import display_and_ocr
from modules.llm import (
    classify_themes_with_llm,
    analyze_themes_with_gpt,
    context_analysis,
    summarize_text,
    generate_image_from_summary,
    summarize_text_sound,
    summarize_language
)
from modules.audio import handle_tts
from modules.utils import get_base64_image
from styles.custom_css import inject_custom_css

from config import (
    TUMO_LOGO_PATH,
    CALFA_LOGO_PATH,
    NLA_LOGO_PATH
)

# Inject CSS
inject_custom_css()

# Load models and OpenAI client
detection_model, classification_model, rotation_model = load_yolo_models()
client = load_openai_client()

# Sidebar
st.sidebar.markdown('<div class="sidebar-title">📰 Historical Newspapers</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

st.sidebar.markdown("### ⚙️ Processing Options")
run_quality_check = st.sidebar.checkbox("🧪 Check quality", value=True)
run_rotation_check = st.sidebar.checkbox("🧭 Check Rotation and crop", value=True)
run_ocr = st.sidebar.checkbox("🔠 Perform OCR", value=True)
run_llm = st.sidebar.checkbox("🧠 Perform LLM Classification", value=False)

selected_llm_task = None
if run_llm:
    selected_llm_task = st.sidebar.selectbox(
        "Select LLM Task",
        [
            "Topic modeling",
            "Context Analysis",
            "Summary to image",
            "Text to speech",
            "Text summarization"
        ]
    )

run_button_clicked = st.sidebar.button("🚀 Run")

st.sidebar.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)

# Logos
tumo_logo = get_base64_image(TUMO_LOGO_PATH)
calfa_logo = get_base64_image(CALFA_LOGO_PATH)
nla_logo = get_base64_image(NLA_LOGO_PATH)

st.sidebar.markdown(f"""
    <div style="text-align: center; margin-top: 15px;">
        <img src="data:image/png;base64,{tumo_logo}" width="120"><br><br>
        <img src="data:image/png;base64,{calfa_logo}" width="120"><br><br>
        <img src="data:image/png;base64,{nla_logo}" width="80">
    </div>
""", unsafe_allow_html=True)

# --- Main Content ---
st.title("📰 Newspaper Analyzer")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if run_button_clicked:
        label = None
        if run_quality_check:
            label = classify_quality(image, classification_model)

        if run_rotation_check:
            image = detect_rotation(image, rotation_model)

        if label == "sports_car":
            st.warning("Low quality image detected. Are you sure you want to continue?")
            if st.button("Continue anyway"):
                st.session_state['run_detection'] = True
        else:
            st.session_state['run_detection'] = True

    if st.session_state.get('run_detection', False):
        detected_img, masks = yolo_detect(image, detection_model)
        st.image(detected_img, caption="Detection Output")

        crops, binarized_crops = crop_objects(image, masks)

        if run_ocr and crops:
            extracted_texts = display_and_ocr(crops, binarized_crops)

            if run_llm and selected_llm_task:
                full_text = "\n".join(extracted_texts.values())

                if selected_llm_task == "Topic modeling":
                    themes = classify_themes_with_llm(client, extracted_texts)
                    st.success("Themes Identified:")
                    st.write(themes)

                elif selected_llm_task == "Context Analysis":
                    context = context_analysis(client, extracted_texts)
                    st.success("Context Analysis:")
                    st.write(context)

                elif selected_llm_task == "Summary to image":
                    summary = summarize_text(client, full_text)
                    st.success("Summary Generated:")
                    st.write(summary)

                    image_url = generate_image_from_summary(client, summary)
                    if image_url:
                        st.image(image_url, caption="Generated Image", use_container_width=True)
                    else:
                        st.warning("Failed to generate image.")

                elif selected_llm_task == "Text to speech":
                    summary = summarize_text_sound(client, full_text)
                    audio_path = handle_tts(summary)
                    st.audio(audio_path, format="audio/wav")
                    with open(audio_path, "rb") as audio_file:
                        st.download_button("🔊 Download Speech", audio_file, file_name="speech.wav")

                elif selected_llm_task == "Text summarization":
                    hy_summary = summarize_language(client, full_text, language='hy')
                    fr_summary = summarize_language(client, full_text, language='fr')

                    st.success("Armenian Summary")
                    st.write(hy_summary)

                    st.success("French Summary")
                    st.write(fr_summary)
