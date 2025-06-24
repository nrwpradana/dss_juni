import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import hashlib

# ====== SETUP ======
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="huawei-noah/TinyBERT_General_4L_312D", device=-1)  # Force CPU

embedding_model = load_embedding_model()
qa_model = load_qa_model()

@st.cache_data(persist=True)
def build_faiss_index(texts, _cache_key):
    embeddings = embedding_model.encode(texts, show_progress_bar=False, batch_size=32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product for faster indexing
    faiss.normalize_L2(embeddings)  # Normalize for IP
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

def retrieve(query, index, df, top_k=2):
    query_embedding = embedding_model.encode([query], show_progress_bar=False)[0]
    faiss.normalize_L2(np.array([query_embedding]).astype("float32"))
    distances, indices = index.search(np.array([query_embedding]).astype("float32"), top_k)
    return df.iloc[indices[0]]

def generate_answer(query, context):
    context = " ".join(context.tolist())  # Combine context into single string
    try:
        result = qa_model(question=query, context=context, max_answer_len=100)
        return result["answer"]
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# ====== STREAMLIT UI ======
st.title("📊 Lightweight RAG for CSV")

# ====== SIDEBAR ======
st.sidebar.header("🔧 Settings")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type="csv")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "cache_key" not in st.session_state:
    st.session_state.cache_key = None

# Reset history button
if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history = []
    st.sidebar.success("History cleared!")

# ====== MAIN INPUT ======
if uploaded_file:
    # Generate unique cache key based on file content
    file_content = uploaded_file.getvalue()
    cache_key = hashlib.md5(file_content).hexdigest()
    if st.session_state.cache_key != cache_key:
        st.session_state.cache_key = cache_key
        st.cache_data.clear()  # Clear cache for new file

    # Read CSV with limited rows to reduce memory
    try:
        df = pd.read_csv(uploaded_file, nrows=1000)  # Limit to 1000 rows
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        st.stop()

    st.subheader("🗂️ Select Columns for RAG")
    selected_columns = st.multiselect(
        "Choose columns to process:",
        options=df.columns.tolist(),
        default=df.columns.tolist()
    )

    if not selected_columns:
        st.warning("⚠️ Please select at least one column.")
        st.stop()

    # Show data preview
    st.write("📄 Data Preview")
    st.dataframe(df[selected_columns].head(5))

    def transform_data(df, selected_columns):
        df["text"] = df[selected_columns].astype(str).agg(" | ".join, axis=1)
        return df

    # Query input
    query = st.text_input("❓ Enter Your Question")
    run_query = st.button("🚀 Answer Question")

    if run_query and query:
        try:
            df = transform_data(df, selected_columns)
            index, _ = build_faiss_index(df["text"].tolist(), _cache_key=cache_key)

            with st.spinner("🔍 Retrieving relevant data..."):
                results = retrieve(query, index, df)
                context = results["text"]

            with st.spinner("🧠 Generating answer..."):
                answer = generate_answer(query, context)

            st.subheader("💬 Answer:")
            st.success(answer)
            st.session_state.history.append((query, answer))

        except Exception as e:
            st.error(f"Error: {str(e)}")
    elif run_query and not query:
        st.warning("⚠️ Please enter a question.")
else:
    st.info("📂 Upload a CSV file to start.")

# ====== HISTORY ======
if st.session_state.history:
    st.subheader("🕘 Question and Answer History")
    for i, (q, a) in enumerate(reversed(st.session_state.history[-5:]), 1):
        with st.expander(f"❓ Question #{len(st.session_state.history)-i+1}: {q}"):
            st.markdown(f"💬 **Answer:** {a}")
