import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

# Fungsi untuk memproses data tabular
def process_tabular_data(df):
    documents = []
    for _, row in df.iterrows():
        doc = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        documents.append(doc)
    return documents

# Fungsi untuk indexing dan pencarian
def create_index(documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
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
            repo_id="meta-llama/Llama-3-8b",
            huggingfacehub_api_token=api_token
        )
        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Berdasarkan konteks berikut: {context}\nJawab pertanyaan: {query}"
        )
        response = llm(prompt.format(query=query, context=context))
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit App
st.title("RAG-Enhanced Structured Data Insights")

# Input Hugging Face API token
api_token = st.text_input("Masukkan Hugging Face API Token:", type="password")
if not api_token:
    st.warning("Silakan masukkan Hugging Face API Token untuk melanjutkan.")
    st.stop()

# Upload file CSV
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data yang diunggah:")
    st.dataframe(df)

    # Proses data
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
            answer = generate_answer(query, context, api_token)
        st.write("Jawaban:")
        st.write(answer)
