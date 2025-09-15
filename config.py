import os
import streamlit as st
from dotenv import load_dotenv

# Carrega .env local
load_dotenv()

def get_secret(key):
    """
    Retorna o segredo da chave fornecida.
    - Se estiver no Streamlit Cloud, usa st.secrets
    - Senão, usa variáveis de ambiente local
    """
    if "STREAMLIT_SERVER" in os.environ:  # verifica se está no Streamlit Cloud
        return st.secrets.get(key)
    else:
        return os.getenv(key)
