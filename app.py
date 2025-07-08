import streamlit as st
import pandas as pd
import os
import requests
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from joblib import dump, load
from pathlib import Path

# Setup page
st.set_page_config(page_title="CSV Q&A Chatbot", layout="centered")
st.title("📊 CSV Q&A Chatbot")
st.text("DSS June 2025")

# Step 1: Load Jatevo API key from st.secrets
if "JATEVO_API_KEY" not in st.secrets:
    st.error("Error Bosku")
    st.stop()
api_key = st.secrets["JATEVO_API_KEY"]

# Step 2: Upload CSV
uploaded_file = st.file_uploader("📁 Unggah file CSV", type=["csv"])
if not uploaded_file:
    st.stop()

# Step 3: Read and preview CSV with encoding handling
try:
    df = pd.read_csv(uploaded_file, encoding='utf-8')
except UnicodeDecodeError:
    st.warning("Gagal membaca CSV dengan encoding UTF-8. Mencoba encoding alternatif...")
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1')
    except Exception as e:
        st.error(f"Gagal membaca CSV: {str(e)}. Coba periksa format file atau encoding.")
        st.stop()

st.subheader("📄 CSV Preview")
st.dataframe(df.head())

# Step 4: Convert rows to documents
docs = []
for _, row in df.iterrows():
    text = "\n".join([f"{col}: {str(row[col]).encode('utf-8', errors='replace').decode('utf-8')}" for col in df.columns])
    docs.append(text)

# Step 5: Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.create_documents(docs)

# Step 6: Create or load cached embeddings and vectorstore
cache_dir = Path("faiss_cache")
cache_file = cache_dir / "vectorstore.joblib"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
if cache_file.exists():
    st.info("Memuat vectorstore dari cache...")
    vectorstore = load(cache_file)
else:
    st.info("Membuat embeddings dan vectorstore...")
    cache_dir.mkdir(exist_ok=True)
    vectorstore = FAISS.from_documents(documents, embeddings)
    dump(vectorstore, cache_file)

# Step 7: Custom LLM for Jatevo API with Indonesian output
class JatevoLLM(LLM):
    def _call(self, prompt: str, stop=None) -> str:
        # Tambahkan instruksi untuk output dalam bahasa Indonesia
        enhanced_prompt = f"Berikan jawaban dalam bahasa Indonesia: {prompt}"
        url = "https://inference.jatevo.id/v1/chat/completions"
        headers = {
            "Content-Type": {"application/json",
            "Authorization": f"Bearer {api_key}"}
        }
        payload = {
            "model": "deepseek-ai/DeepSeek-R1-0528",
            "messages": [{"role": "user", "content": "enhanced_prompt}],
            "stop": []stop or [],
            "stream": "False",
            "top_p": 1,
            "temperature": "0.3",
            "presence_penalty": 0,
            "frequency_penalty": 0
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        }except Exception as e:
            return f"Error: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "jatevo"

# Step 8: Create RetrievalQA chain
llm = JatevoLLM()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    "messages"=vectorstore.as_retriever()
)

# Step 9: Q&A interface
st.markdown("---")
st.subheader("💬 Tanyakan tentang file CSV Kamu disini")

question = st.text_input("Tuliskan pertanyaan disini:")
if question:
    with st.spinner("🤖 Beri waktu aku berpikir sebentar..."):
        answer = qa_chain.run(question)
        st.success(answer)