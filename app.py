import streamlit as st
import pandas as pd
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

st.set_page_config(page_title="CSV Q&A Chatbot", layout="centered")
st.title("📊 CSV Q&A Chatbot with RAG + Hugging Face")

# Step 1: Ask for API token securely
api_token = st.text_input("🔐 Enter your Hugging Face API token:", type="password")
if not api_token:
    st.warning("Please enter your Hugging Face API token to proceed.")
    st.stop()

# Set environment variable dynamically
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

# Step 2: Upload CSV
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Preview of uploaded CSV:")
    st.dataframe(df.head())

    # Step 3: Convert CSV rows into plain text documents
    docs = []
    for _, row in df.iterrows():
        text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(text)

    # Step 4: Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.create_documents(docs)

    # Step 5: Embed and index with FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Step 6: Setup LLM with Hugging Face
    llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
    )

    # Step 7: QA chain using vectorstore retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # Step 8: User Q&A
    st.markdown("---")
    st.subheader("💬 Ask a question about your CSV:")
    question = st.text_input("Type your question:")

    if question:
        with st.spinner("🔍 Thinking..."):
            answer = qa_chain.run(question)
            st.success(answer)
