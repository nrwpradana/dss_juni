import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import T5ForConditionalGeneration, T5Tokenizer
import chardet
import torch
import logging

# Setup logging untuk debugging di Streamlit Cloud
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
        return 'utf-8'

# Fungsi untuk membaca CSV dengan penanganan encoding
def read_csv_with_encoding(file, encoding_choice):
    try:
        if encoding_choice != "auto":
            return pd.read_csv(file, encoding=encoding_choice)
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
        logger.error(f"Error reading CSV: {str(e)}")
        st.error(f"Error membaca file CSV: {str(e)}")
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
        logger.error(f"Error processing tabular data: {str(e)}")
        st.error(f"Error processing data: {str(e)}")
        return []

# Fungsi untuk indexing dan pencarian
@st.cache_resource
def create_index(_documents):
    try:
        model = SentenceTransformer('distilbert-base-nli-mean-tokens', device='cpu')
        embeddings = model.encode(_documents, show_progress_bar=False, batch_size=16)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, model, embeddings
    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")
        st.error(f"Error creating index: {str(e)}")
        return None, None, None

# Fungsi untuk mencari dokumen relevan
def search_documents(query, index, model, documents, k=3):
    try:
        query_embedding = model.encode([query], show_progress_bar=False)
        distances, indices = index.search(query_embedding, k)
        return [(documents[i], distances[0][j]) for j, i in enumerate(indices[0])]
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        st.error(f"Error searching documents: {str(e)}")
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
        logger.error(f"Error loading T5 model: {str(e)}")
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Fungsi untuk menghasilkan jawaban dengan T5 lokal
def generate_answer(query, context, model, tokenizer):
    try:
        prompt = f"Berdasarkan konteks berikut: {context}\nJawab pertanyaan: {query}"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
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
    with st.spinner("Memproses file CSV..."):
        df = read_csv_with_encoding(uploaded_file, encoding_choice)
    
    if df is not None:
        st.write("Data yang diunggah:")
        st.dataframe(df)

        # Proses data
        with st.spinner("Memproses data..."):
            documents = process_tabular_data(df)
            if not documents:
                st.stop()
            index, model, embeddings = create_index(documents)
            if index is None:
                st.stop()

        # Input query
        query = st.text_input("Masukkan pertanyaan Anda:")
        if query:
            # Cari dokumen relevan
            with st.spinner("Mencari dokumen relevan..."):
                results = search_documents(query, index, model, documents)
                if not results:
                    st.stop()
                st.write("Dokumen relevan:")
                for doc, score in results:
                    st.write(f"Skor: {score Facade: