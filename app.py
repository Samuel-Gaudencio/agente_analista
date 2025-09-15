import gspread
import pandas as pd
import streamlit as st
from oauth2client.service_account import ServiceAccountCredentials
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
import google.generativeai as palm
from config import get_secret

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
CREDENCIAIS_JSON = get_secret("GOOGLE_CREDENTIALS_JSON")

# =========================
# CONFIGURAÇÃO GEMINI
# =========================
palm.configure(api_key=GEMINI_API_KEY)

# =========================
# 1. Conexão com Google Sheets
# =========================
def carregar_planilha(nome_planilha: str, aba: str, credenciais_json: str):
    # Escopo necessário para acessar Google Sheets
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    # Detecta se está no Streamlit Cloud
    if "google_credentials" in st.secrets:
        # Pega as credenciais direto do secrets
        credenciais_dict = dict(st.secrets["google_credentials"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(credenciais_dict, scope)
    else:
        # Usa arquivo local .json quando rodando localmente
        creds = ServiceAccountCredentials.from_json_keyfile_name(credenciais_json, scope)

    # Autentica e abre a planilha
    cliente = gspread.authorize(creds)
    planilha = cliente.open(nome_planilha)
    worksheet = planilha.worksheet(aba)

    # Converte os dados em dataframe
    dados = worksheet.get_all_records()
    import pandas as pd
    df = pd.DataFrame(dados)
    return df


# =========================
# 2. Transformar em documentos (blocos de 50 linhas)
# =========================
def criar_documentos(df, chunk_size=50):
    documents = []
    for i in range(0, len(df), chunk_size):
        bloco = df.iloc[i:i+chunk_size]
        content = "\n".join(
            [", ".join([f"{k}: {v}" for k, v in row.items()]) for _, row in bloco.iterrows()]
        )
        doc = Document(page_content=content, metadata={"inicio": i, "source": "GoogleSheet"})
        documents.append(doc)
    return documents


# =========================
# 3. Wrapper customizado Gemini
# =========================
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [
            palm.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )["embedding"]
            for text in texts
        ]

    def embed_query(self, text):
        embedding = palm.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="RETRIEVAL_QUERY"
        )["embedding"]
        return embedding


# =========================
# 4. Criar vetor de embeddings (Chroma)
# =========================
def criar_vectorstore(documents):
    embeddings = GeminiEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


# =========================
# 5. Criar agente RAG (PROMPT EM PORTUGUÊS)
# =========================
def criar_agente(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.0, google_api_key=GEMINI_API_KEY)

    prompt_template = """Você é um Analista de Dados Sênior, especialista em transformar informações brutas em respostas claras, precisas e embasadas.
Seu objetivo é analisar os dados fornecidos em <contexto> e responder à pergunta em <pergunta> de forma objetiva, profissional e exclusivamente em português.

<regras>
- Utilize apenas as informações presentes em `<contexto>`.
- Caso a resposta não esteja disponível, responda exatamente: **"Essa informação não foi encontrada nos dados."**
- Mantenha a comunicação em tom técnico, mas simples e direto, sem jargões desnecessários.
- Quando relevante, apresente a resposta estruturada em tópicos, listas ou tabelas para maior clareza.
- Evite inferências que não possam ser sustentadas pelos dados fornecidos.
- Se houver números, estatísticas ou métricas, destaque-os de forma clara.
</regras>

Contexto:
{context}

Pergunta: {question}
Resposta:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 30}),
        chain_type_kwargs=chain_type_kwargs
    )
    return qa


# =========================
# 6. Função principal Streamlit
# =========================
def main():
    st.set_page_config(page_title="Agente de Dados AI", layout="wide")
    st.title("📊 Agente de Dados Sênior (Gemini)")

    # ---- Carrega dados e configura agente (cache para não refazer toda hora)
    @st.cache_resource(show_spinner="Carregando dados e criando embeddings...")
    def iniciar():
        df = carregar_planilha("base", "Dados", CREDENCIAIS_JSON)
        documents = criar_documentos(df)
        vectorstore = criar_vectorstore(documents)
        qa = criar_agente(vectorstore)
        return df, qa

    df, qa = iniciar()

    # ---- Sidebar de navegação
    pagina = st.sidebar.radio("Escolha uma página", ["Chat com o Agente", "Visualizar Planilha"])

    if pagina == "Chat com o Agente":
        st.subheader("💬 Pergunte ao agente")
        pergunta = st.text_input("Digite sua pergunta:")

        if st.button("Enviar"):
            if pergunta:
                with st.spinner("O agente está processando..."):
                    resposta = qa.invoke({"query": pergunta})
                    st.markdown("### 🧠 Resposta do Agente:")
                    st.write(resposta['result'])
            else:
                st.warning("Digite uma pergunta antes de enviar.")

    elif pagina == "Visualizar Planilha":
        st.subheader("📋 Dados da Planilha")
        st.dataframe(df)
        st.markdown(f"**Total de linhas:** {len(df)}")


if __name__ == "__main__":
    main()


