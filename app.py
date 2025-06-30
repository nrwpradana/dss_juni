import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import T5ForConditionalGeneration, T5Tokenizer
import chardet
import torch
import logging

# Setup logging untuk debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fungsi untuk mendeteksi encoding file
def detect_encoding(file):
    try:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        file.seek(0)
        return result['encoding']
    except Exception as e:
        logger.error(f"Error detecting encoding: {str(e)}")
        return None

# Fungsi untuk membaca CSV dengan penanganan encoding
def read_csv_with_encoding(file, encoding_choice):
    try:
        if encoding_choice != "auto":
            return pd.read_csv(file, encoding=encoding_choice)
        encoding = detect_encoding(file)
        if encoding:
            logger.info(f"Detected encoding: {encoding}")
            st.write(f"Detected encoding: {encoding}")
            return pd.read_csv(file, encoding=encoding)
        encodings = ['latin1', 'iso-8859-1', 'windows-1252']
        for enc in encodings:
            try:
                file.seek(0)
                return pd.read_csv(file, encoding=enc)
            except UnicodeDecodeError:
                continue
        st.error("Gagal membaca file CSV. Pastikan file menggunakan encoding yang valid (UTF-8, Latin1, dll.).")
        logger.error("Failed to read CSV with all encodings")
        return None
    except Exception as e:
        st.error(f"Error membaca file CSV: {str(e)}")
        logger.error(f"Error reading CSV: {str(e)}")
        return None

# Fungsi untuk memproses data tabular
@st.cache_data
def process_tabular_data(_df):
    try:
        documents = []
        for _, row in _df.iterrows():
            doc = ", ".join([f"{col}: {row[col]}" for col in _df.columns])
            documents.append(doc)
        return documents
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        logger.error(f"Error processing data: {str(e)}")
        return None

# Fungsi untuk indexing dan pencarian
@st.cache_resource
def create_index(_documents):
    try:
        model = SentenceTransformer('distilbert-base-multilingual-cased')
        embeddings = model.encode(_documents, show_progress_bar=False)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, model, embeddings
    except Exception as e:
        st.error(f"Error creating index: {str(e)}")
        logger.error(f"Error creating index: {str(e)}")
        return None, None, None

# Fungsi untuk mencari dokumen relevan
def search_documents(query, index, model, documents, k=3):
    try:
        query_embedding = model.encode([query])
        distances, indices = index.search(query_embedding, k)
        return [(documents[i], distances[0][j]) for j, i in enumerate(indices[0])]
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        logger.error(f"Error searching documents: {str(e)}")
        return []

# Fungsi untuk memuat model T5 lokal
@st.cache_resource
def load_t5_model():
    try:
        model_name = "google/flan-t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading T5 model: {str(e)}")
        logger.error(f"Error loading T5 model: {str(e)}")
        return None, None

# Fungsi untuk menghasilkan jawaban dengan T5 lokal
def generate_answer(query, context, model, tokenizer):
    try:
        input_text = f"Berdasarkan konteks berikut: {context}\nJawab pertanyaan: {query}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        logger.error(f"Error generating answer: {str(e)}")
        return f"Error generating answer: {str(e)}"

# Streamlit App
st.title("RAG-Enhanced Structured Data Insights")

# Sidebar untuk input
with st.sidebar:
    st.header("Konfigurasi")
    # Upload file CSV
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    # Opsi encoding manual
    encoding_choice = st.selectbox("Pilih encoding CSV (opsional):", ["auto", "utf-8", "latin1", "iso-8859-1", "windows-1252"])

# Main content
if uploaded_file:
    df = read_csv_with_encoding(uploaded_file, encoding_choice)
    if df is not None:
        st.write("Data yang diunggah:")
        st.dataframe(df)

        # Proses data
        with st.spinner("Memproses data..."):
            documents = process_tabular_data(df)
            if documents is None:
                st.stop()
            index, model, embeddings = create_index(documents)
            if index is None:
                st.stop()

        # Input query
        query = st.text_input("Masukkan pertanyaan Anda:")
        if query:
            # Cari dokumen relevan
            results = search_documents(query, index, model, documents)
            if results:
                st.write("Dokumen relevan:")
                for doc, score in results:
                    st.write(f"Skor: {score:.4f} | {doc}")

                # Gabungkan konteks untuk LLM
                context = "\n".join([doc for doc, _ in results])
                with st.spinner("Menghasilkan jawaban..."):
                    t5_model, t5_tokenizer = load_t5_model()
                    if t5_model is None:
                        st.stop()
                    answer = generate_answer(query, context, t5_model, t5_tokenizer)
                    st.write("Jawaban:")
                    st.write(answer)
else:
    st.info("Silakan unggah file CSV di sidebar untuk memulai.")