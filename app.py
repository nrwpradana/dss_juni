import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain.prompts import PromptTemplate
import chardet
import torch

# Fungsi untuk mendeteksi encoding file
def detect_encoding(file):
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file.seek(0)
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

# Fungsi untuk memproses data tabular
@st.cache_data
def process_tabular_data(_df):
    documents = []
    for _, row in _df.iterrows():
        doc = ", ".join([f"{col}: {row[col]}" for col in _df.columns])
        documents.append(doc)
    return documents

# Fungsi untuk indexing dan pencarian
@st.cache_resource
def create_index(_documents):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(_documents, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, model, embeddings

# Fungsi untuk mencari dokumen relevan
def search_documents(query, index, model, documents, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [(documents[i], distances[0][j]) for j, i in enumerate(indices[0])]

# Fungsi untuk memuat model T5 lokal
@st.cache_resource
def load_t5_model():
    model_name = "google/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

# Fungsi untuk menghasilkan jawaban dengan T5 lokal
def generate_answer(query, context, model, tokenizer):
    try:
        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Berdasarkan konteks berikut: {context}\nJawab pertanyaan: {query}"
        )
        input_text = prompt.format(query=query, context=context)
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
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
    if encoding_choice != "auto":
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding_choice)
        except Exception as e:
            st.error(f"Error membaca file CSV dengan encoding {encoding_choice}: {str(e)}")
            df = None
    else:
        df = read_csv_with_encoding(uploaded_file)
    
    if df is not None:
        st.write("Data yang diunggah:")
        st.dataframe(df)

        # Proses data
        with st.spinner("Memproses data..."):
            documents = process_tabular_data(df)
            index, model, embeddings = create_index(documents)

        # Input query
        query = st.text_input("Masukkan pertanyaan Anda:")
        if query:
            # Cari dokumen relevan
            results = search_documents(query, index, model, documents)
            st.write("Dokumen relevan:")
            for doc, score in results:
                st.write(f"Skor: {score:.4f} | {doc}")

            # Gabungkan konteks untuk LLM
            context = "\n".join([doc for doc, _ in results])
            with st.spinner("Menghasilkan jawaban..."):
                t5_model, t5_tokenizer = load_t5_model()
                answer = generate_answer(query, context, t5_model, t5_tokenizer)
                st.write("Jawaban:")
                st.write(answer)
else:
    st.info("Silakan unggah file CSV di sidebar untuk memulai.")