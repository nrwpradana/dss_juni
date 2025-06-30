import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
import chardet
import os

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
        df = pd.read_csv(file, encoding=encoding)
        # Validasi CSV
        if df.empty:
            st.error("File CSV kosong. Silakan unggah file dengan data.")
            return None
        if df.columns.empty:
            st.error("File CSV tidak memiliki kolom. Pastikan format CSV valid.")
            return None
        return df
    except UnicodeDecodeError:
        encodings = ['latin1', 'iso-8859-1', 'windows-1252']
        for enc in encodings:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc)
                if df.empty or df.columns.empty:
                    st.error("File CSV kosong atau tidak memiliki kolom.")
                    return None
                return df
            except UnicodeDecodeError:
                continue
        st.error("Gagal membaca file CSV. Pastikan file menggunakan encoding yang valid (UTF-8, Latin1, dll.).")
        return None
    except Exception as e:
        st.error(f"Error membaca file CSV: {str(e)}")
        return None

# Fungsi untuk memproses data tabular
def process_tabular_data(df):
    documents = []
    for idx, row in df.iterrows():
        # Format dokumen dengan indeks baris untuk pelacakan
        doc = f"Row {idx}: " + ", ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
        documents.append(doc)
    return documents

# Fungsi untuk indexing dan pencarian
def create_index(documents):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(documents, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, model, embeddings
    except Exception as e:
        st.error(f"Error creating index: {str(e)}")
        return None, None, None

# Fungsi untuk mencari dokumen relevan
def search_documents(query, index, model, documents, k=5):
    try:
        query_embedding = model.encode([query])
        distances, indices = index.search(query_embedding, k)
        return [(documents[i], distances[0][j]) for j, i in enumerate(indices[0])]
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return []

# Fungsi untuk menghasilkan jawaban dengan LLM
def generate_answer(query, context, api_token, model_id="meta-llama/Meta-Llama-3-8B"):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            huggingfacehub_api_token=api_token,
            max_new_tokens=512,
            temperature=0.7
        )
        # Prompt yang lebih spesifik untuk data tabular
        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "Kamu adalah asisten analisis data yang menjawab pertanyaan berdasarkan data dari file CSV. "
                "Konteks berikut berisi baris-baris relevan dari CSV:\n\n{context}\n\n"
                "Pertanyaan: {query}\n\n"
                "Jawab pertanyaan secara akurat berdasarkan data dalam konteks. Jika data tidak cukup, "
                "katakan bahwa informasi tidak tersedia dan jelaskan mengapa. Gunakan bahasa yang jelas dan ringkas."
            )
        )
        response = llm.invoke(prompt.format(query=query, context=context))
        return response
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return None

# Streamlit App
st.title("RAG-Enhanced Structured Data Insights")

# Sidebar untuk input
with st.sidebar:
    st.header("Konfigurasi")
    # Input Hugging Face API token
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.text_input("Masukkan Hugging Face API Token:", type="password")
    if not api_token:
        st.warning("Silakan masukkan Hugging Face API Token di sidebar atau setel sebagai variabel lingkungan HUGGINGFACEHUB_API_TOKEN.")
    
    # Pilih model
    model_id = st.selectbox("Pilih Model:", ["meta-llama/Meta-Llama-3-8B", "google/flan-t5-base"])
    if model_id != "meta-llama/Meta-Llama-3-8B":
        st.warning(f"Model {model_id} mungkin memiliki performa lebih rendah untuk tugas kompleks dibandingkan Meta-Llama-3-8B.")

    # Upload file CSV
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    
    # Parameter pencarian
    k_docs = st.slider("Jumlah dokumen relevan yang diambil:", 1, 10, 5)

# Main content
if uploaded_file:
    df = read_csv_with_encoding(uploaded_file)
    if df is not None:
        st.write("Data yang diunggah:")
        st.dataframe(df)

        # Tampilkan pratinjau dokumen yang diproses (untuk debugging)
        documents = process_tabular_data(df)
        st.write("Dokumen yang diproses (pratinjau 5 baris pertama):")
        for doc in documents[:5]:
            st.write(doc)

        # Buat indeks
        index, model, embeddings = create_index(documents)
        if index is None:
            st.error("Gagal membuat indeks. Periksa data atau model embedding.")
        else:
            # Input query
            query = st.text_input("Masukkan pertanyaan Anda:")
            if query:
                if not api_token:
                    st.error("Harap masukkan Hugging Face API Token di sidebar atau setel sebagai variabel lingkungan.")
                else:
                    # Debug info
                    st.write(f"Using model: {model_id}")
                    st.write(f"API token (partial): {api_token[:5]}...")

                    # Cari dokumen relevan
                    results = search_documents(query, index, model, documents, k=k_docs)
                    if not results:
                        st.error("Tidak ada dokumen relevan ditemukan. Coba ubah pertanyaan atau periksa data.")
                    else:
                        st.write("Dokumen relevan:")
                        for doc, score in results:
                            st.write(f"Skor: {score:.4f} | {doc}")

                        # Gabungkan konteks untuk LLM
                        context = "\n".join([doc for doc, _ in results])
                        st.write("Konteks yang dikirim ke LLM:")
                        st.write(context)

                        with st.spinner("Menghasilkan jawaban..."):
                            answer = generate_answer(query, context, api_token, model_id)
                        if answer:
                            st.write("Jawaban:")
                            st.write(answer)
else:
    st.info("Silakan unggah file CSV di sidebar untuk memulai.")