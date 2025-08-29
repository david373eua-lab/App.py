# app.py
import os
import subprocess
import streamlit as st
from joblib import load

MODEL_PATH = os.path.join("models", "modelo_spam.pkl")

# Configuração da página
st.set_page_config(
    page_title="Lumen5 - Classificador de Texto",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Estilo customizado com CSS inline
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    h1 {
        color: #4B0082 !important;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton button {
        background: linear-gradient(90deg, #4B0082, #6A5ACD);
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.6rem 1.2rem;
    }
    .stButton button:hover {
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título
st.title("✨ Lumen5 – Detector de Spam em Mensagens")

# Verificação e treinamento do modelo
if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ Modelo não encontrado. Treinando agora, aguarde...")
    subprocess.run(["python", "train.py"])

if os.path.exists(MODEL_PATH):
    pipe = load(MODEL_PATH)

    with st.form("classificacao"):
        texto = st.text_area("✍️ Digite sua mensagem:", "", height=150)
        submitted = st.form_submit_button("🚀 Classificar")

    if submitted and texto.strip():
        rotulo = pipe.predict([texto])[0]
        if rotulo == "spam":
            st.error("🚨 Essa mensagem parece **SPAM**!")
        else:
            st.success("✅ Essa mensagem parece **Normal (ham)**.")

# Rodapé estilizado
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Feito com 💜 usando Streamlit · Projeto Lumen5</p>",
    unsafe_allow_html=True
)
