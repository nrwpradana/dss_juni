import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import chardet

# Fungsi untuk mendeteksi encoding file
def detect_encoding(file):
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file.seek(0)  # Reset file pointer
    return result['encoding']

# Fungsi untuk membaca CSV dengan penanganan encoding
def read_csv_with_encoding(file):
    try:
        encoding = detect_encoding(file)
        st.write(f"Detected encoding: {encoding}")
        return pd.read_csv(file, encoding=encoding)
    except UnicodeDecodeError:
        encodings = ['latin1', 'iso-8859-1', 'windows-1252']
        for enc in encodings:
            try:
                file.seek(0)
                return pd.read_csv(file, encoding=enc)
            except UnicodeDecodeError:
                continue
        st.error("Gagal membaca file CSV. Pastikan file menggunakan encoding yang valid (UTF-8, Latin1, dll.).")
        return None
    except Exception as e:
        st.error(f"Error membaca file CSV: {str(e)}")
        return None

# Fungsi untuk memvalidasi token API
def validate_api_token(api_token):
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=api_token)
        api.list_models(limit=1)  # Coba panggil API sederhana
        return True
    except Exception as e:
        st.error(f"Invalid Hugging Face API token: {str(e)}")
        return False

# Fungsi untuk memproses data tabular
def process_tabular_data(df):
    documents = []
    for _, row in df.iterrows():
        doc = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        documents.append(doc)
    return documents

# Fungsi untuk indexing dan pencarian
def create_index(documents):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Model multibahasa
    embeddings = model.encode(documents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, model, embeddings

# Fungsi untuk mencari dokumen relevan
def search_documents(query, index, model, documents, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [(documents[i], distances[0][j]) for j, i in enumerate(indices[0])]

# Fungsi untuk menghasilkan jawaban dengan LLM
def generate_answer(query, context, api_token):
    try:
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",  # Model ringan yang mendukung bahasa Indonesia
            huggingfacehub_api_token=api_token,
            model_kwargs={"max_length": 512, "temperature": 0.7}
        )
        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Berdasarkan konteks berikut: {context}\nJawab pertanyaan: {query}"
        )
        response = llm(prompt.format(query=query, context=context))
        return response
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Streamlit App
st.title("RAG-Enhanced Structured Data Insights")

# Sidebar untuk input
with st.sidebar:
    st.header("Konfigurasi")
    # Input Hugging Face API token
    api_token = st.text_input("Masukkan Hugging Face API Token:", type="password")
    if api_token and not validate_api_token(api_token):
        st.stop()

    # Upload file CSV
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

# Main content
if uploaded_file:
    df = read_csv_with_encoding(uploaded_file)
    if df is not None:
        st.write("Data yang diunggah:")
        st.dataframe(df)

        # Proses data
        documents = process_tabular_data(df)
        index, model, embeddings = create_index(documents)

        # Input query
        query = st.text_input("Masukkan pertanyaan Anda:")
        if query:
            if not api_token:
                st.error("Harap masukkan Hugging Face API Token di sidebar.")
            else:
                # Cari dokumen relevan
                results = search_documents(query, index, model, documents)
                st.write("Dokumen relevan:")
                for doc, score in results:
                    st.write(f"Skor: {score:.4f} | {doc}")

                # Gabungkan konteks untuk LLM
                context = "\n".join([doc for doc, _ in results])
                with st.spinner("Menghasilkan jawaban..."):
                    answer = generate_answer(query, context, api_token)
                st.write("Jawaban:")
                st.write(answer)
else:
    st.info("Silakan unggah file CSV di sidebar untuk memulai.")