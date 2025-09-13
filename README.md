# Agente de Dados Sênior AI (Gemini)

Um agente de análise de dados que lê informações de uma planilha do Google Sheets e responde perguntas usando a IA **Gemini** da Google. Interface web feita com **Streamlit**.

---

## Funcionalidades

- Carrega dados de uma planilha do Google Sheets.
- Cria embeddings com Gemini para pesquisa semântica (RAG).
- Responde perguntas de forma estruturada e em português.
- Interface web via Streamlit:
  - Chat com o agente.
  - Visualização da planilha.

---

## Tecnologias

- Python 3.10+
- LangChain
- Google Gemini API
- Streamlit
- Google Sheets API via gspread
- python-dotenv

---

## Instalação

1. Clone o repositório:

``bash
git clone https://github.com/seu-usuario/agente-dados-gemini.git
cd agente-dados-gemini``

2. Crie e ative um ambiente virtual:

``
python -m venv .venv
.venv\Scripts\activate``

3. Instale as dependências:

``pip install -r requirements.txt``

4. Configure suas chaves:

- Local (.env):

``GEMINI_API_KEY="sua_chave_gemini"``
``GOOGLE_CREDENTIALS_JSON="credentials.json"``

- Streamlit Cloud (Secrets):

``GEMINI_API_KEY="sua_chave_gemini"``
``GOOGLE_CREDENTIALS_JSON='{
  "type": "service_account",
  "project_id": "...",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "...",
  "client_id": "...",
  "auth_uri": "...",
  "token_uri": "...",
  "auth_provider_x509_cert_url": "...",
  "client_x509_cert_url": "..."
}'``

## Uso

Execute o app com Streamlit:
``streamlit run app.py``


## Licença

Este projeto é de uso pessoal.

