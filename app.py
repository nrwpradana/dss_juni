import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# ====== SETUP ======
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")  # Lighter than L6-v2

@st.cache_resource
def load_llm_model():
    # Load a lightweight DistilBERT model for question answering
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

sentence_model = load_sentence_model()
qa_model = load_llm_model()

def build_faiss_index(texts):
    embeddings = sentence_model.encode(texts, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

def retrieve(query, index, df, top_k=3):  # Reduced top_k for faster processing
    query_embedding = sentence_model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]).astype("float32"), top_k)
    return df.iloc[indices[0]]

def generate_answer(query, context):
    # Combine retrieved texts into a single context string
    context = "\n".join(context.tolist())
    # Use DistilBERT for question answering
    result = qa_model(question=query, context=context)
    return result["answer"]

# ====== STREAMLIT UI ======
st.title("📊 RAG CSV Umum (Ringan)")

# ====== SIDEBAR ======
st.sidebar.header("🔧 Pengaturan")

uploaded_file = st.sidebar.file_uploader("📂 Upload file CSV", type="csv")

# Inisialisasi session state
if "history" not in st.session_state:
    st.session_state.history = []

# Tombol reset riwayat
if st.sidebar.button("🗑️ Hapus Riwayat"):
    st.session_state.history = []
    st.sidebar.success("Riwayat berhasil dihapus!")

# ====== MAIN INPUT ======
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🗂️ Pilih Kolom untuk Dipelajari oleh RAG")
    selected_columns = st.multiselect(
        "Pilih kolom yang ingin diproses:",
        options=df.columns.tolist(),
        default=df.columns.tolist()
    )

    if not selected_columns:
        st.warning("⚠️ Harap pilih setidaknya satu kolom.")
        st.stop()

    # Tampilkan preview data dari kolom terpilih
    st.write("📄 Pratinjau Data Terpilih")
    st.dataframe(df[selected_columns])

    def transform_data(df, selected_columns):
        df["text"] = df[selected_columns].astype(str).agg(" | ".join, axis=1)
        return df

    # Input pertanyaan
    query = st.text_input("❓ Masukkan pertanyaan Anda")
    run_query = st.button("🚀 Jawab Pertanyaan")

    if run_query:
        try:
            df = transform_data(df, selected_columns)
            index, _ = build_faiss_index(df["text"].tolist())

            with st.spinner("🔍 Mencari data relevan..."):
                results = retrieve(query, index, df)
                context = results["text"]

            with st.spinner("🧠 Menghasilkan jawaban..."):
                answer = generate_answer(query, context)

            st.subheader("💬 Jawaban:")
            st.success(answer)
            st.session_state.history.append((query, answer))

        except Exception as e:
            st.error(f"Terjadi error: {str(e)}")
else:
    st.warning("📂 Silakan upload file CSV terlebih dahulu.")

# ====== HISTORY ======
if st.session_state.history:
    st.subheader("🕘 Riwayat Pertanyaan dan Jawaban")
    for i, (q, a) in enumerate(reversed(st.session_state.history[-5:]), 1):
        with st.expander(f"❓ Pertanyaan #{len(st.session_state.history)-i+1}: {q}"):
            st.markdown(f"💬 **Jawaban:** {a}")
