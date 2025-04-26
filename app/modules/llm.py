import json
import re
import streamlit as st

DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"

def _call_openai(client, messages, model=DEFAULT_MODEL, temperature=0.5, spinner_text=None):
    try:
        if spinner_text:
            with st.spinner(spinner_text):
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None


def summarize_language(client, text, language):
    prompts = {
        'fr': "Summarize the following text in French.",
        'hy': "Summarize the following text in Armenian.",
        'en': "Summarize the following text in English."
    }
    prompt_lang = prompts.get(language, prompts['en'])
    prompt = f"""
    You are a summarizer with expertise in recognizing important topics and keywords in the text. {prompt_lang}
    Pay attention to **capitalized words**, **keywords**, and topic names such as "politics", "science", "history", etc.

    Text:
    {text}
    """

    messages = [
        {"role": "system", "content": "You are an intelligent assistant for summarization."},
        {"role": "user", "content": prompt}
    ]
    return _call_openai(client, messages, temperature=0.7, spinner_text="📝 Summarizing language...")

def summarize_text(client, text):
    prompt = (
        'You are an expert summarizer. Your task is to summarize OCR text with possible noise or errors. '
        'Focus on main themes, ignore mistakes, and produce a short coherent summary.'
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]
    return _call_openai(client, messages, temperature=0.5, spinner_text="📝 Summarizing text...")

def summarize_text_sound(client, text):
    prompt = (
        "Correct and summarize noisy OCR text in clear Armenian. Focus on extracting main ideas and key details."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]
    return _call_openai(client, messages, temperature=0.3, spinner_text="🧠 Summarizing for speech...")

def context_analysis(client, texts_dict):
    text = "\n".join(texts_dict.values())
    prompt = (
        "Analyze Armenian OCR newspaper text to infer general context and topics. "
        "Select from: 'politics', 'sports', 'international', 'history', 'science','weather','economics','philosophy','biography'. "
        "Return a JSON array (1 to 4 topics) and a short context if possible."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]
    return _call_openai(client, messages, temperature=0.1, spinner_text="📚 Analyzing context...")

def analyze_themes_with_gpt(client, texts_dict):
    text = "\n".join(texts_dict.values())
    prompt = (
        "Determine the main themes of a noisy Armenian OCR newspaper page. "
        "Select from: 'politic', 'economy', 'war', 'science', 'histoire', 'events', 'culture', 'international'. "
        "Return a JSON array (1 to 4 themes)."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]
    return _call_openai(client, messages, temperature=0, spinner_text="📚 Analyzing themes...")

def classify_themes_with_llm(client, texts_dict):
    with st.spinner("🔍 Classifying themes with LLM..."):
        raw_output = analyze_themes_with_gpt(client, texts_dict)
        if not raw_output:
            st.error("❌ LLM classification failed")
            return []

        try:
            clean_output = re.sub(r"```json|```", "", raw_output).strip()
            themes = json.loads(clean_output)
            st.success("✅ Themes identified")
            st.markdown("### 🏷️ Detected Themes:")
            for theme in themes:
                st.markdown(
                    f"<span style='background-color:#eef;border-radius:4px;padding:6px 12px;margin:4px;display:inline-block'>{theme}</span>",
                    unsafe_allow_html=True
                )
            return themes
        except Exception as e:
            st.error(f"❌ Parsing error: {e}")
            st.text("Raw GPT output:")
            st.text(raw_output)
            return []

def generate_image_from_summary(client, summary):
    """Generate an image based on a text summary using DALL-E."""
    try:
        response = client.images.generate(
            prompt=summary,
            n=1,
            size="512x512"
        )
        return response.data[0].url
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None
