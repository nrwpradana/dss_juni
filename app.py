import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# ====== SETUP ======
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

@st.cache_resource
def load_llm_model():
    # Download the Sahabatai model
    model_path = hf_hub_download(
        repo_id="gmonsoon/gemma2-9b-cpt-sahabatai-v1-instruct-GGUF",
        filename="gemma2-9b-cpt-sahabatai-v1-instruct-Q4_K_M.gguf"
    )
    # Load the model with llama-cpp-python
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,  # Context length
        n_threads=8,  # Adjust based on your CPU
        n_gpu_layers=0  # Set to >0 if using GPU
    )
    return llm

sentence_model = load_sentence_model()
llm_model = load_llm_model()

def build_faiss_index(texts):
    embeddings = sentence_model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

def retrieve(query, index, df, top_k=5):
    query_embedding = sentence_model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]).astype("float32"), top_k)
    return df.iloc[indices[0]]

def generate_answer(query, context):
    system_message = "Kamu adalah asisten cerdas yang menjawab pertanyaan berdasarkan data yang diberikan."
    user_message = f"""
    Pertanyaan: {query}

    Data yang relevan:
    {context}

    Jawab dalam bahasa Indonesia dengan jelas dan ringkas.
    """
    prompt = f"{system_message}\n\n{user_message}"
    
    # Generate response using the Sahabatai model
    response = llm_model(
        prompt=prompt,
        max_tokens=1000,
        temperature=0.7,
        stop=["</s>", "<|eot|>"]  # Adjust based on model's stop tokens
    )
    return response["choices"][0]["text"].strip()

# ====== STREAMLIT UI ======
st.title("📊 RAG CSV Umum (Tanpa Struktur Khusus)")

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
    st.dataframe(df[selected_columns出行

    def transform_data(df, selected_columns):
        df["text"] = df[selected_columns].astype(str).agg(" | ".join, axis=1)
        return df

    # Input pertanyaan hanya muncul jika kolom telah dipilih
    query = st.text_input("❓ Masukkan pertanyaan Anda")
    run_query = st.button("🚀 Jawab Pertanyaan")

    if run_query:
        try:
            df = transform_data(df, selected_columns)
            index, _ = build_faiss_index(df["text"].tolist())

            with st.spinner("🔍 Mencari data relevan..."):
                results = retrieve(query, index, df)
                context = "\n".join(results["text"].tolist())

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
