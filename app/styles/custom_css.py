import streamlit as st

def inject_custom_css():
    st.markdown("""
        <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }
        section[data-testid="stSidebar"] {
            background-color: #f4f6f9;
            border-right: 1px solid #ddd;
        }
        </style>
    """, unsafe_allow_html=True)
