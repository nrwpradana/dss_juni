import streamlit as st
import pandas as pd
import os
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

st.set_page_config(page_title="CSV Q&A Chatbot", layout="centered")
st.title("📊 CSV Q&A Chatbot with RAG + Transformers")

# Step 1: API token input
api_token = st.text_input("🔐 Enter your Hugging Face API token:", type="password")
if not api_token:
    st.warning("Please enter your Hugging Face API token to proceed.")
    st.stop()

# Step 2: Upload CSV
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Preview of uploaded CSV:")
    st.dataframe(df.head())

    # Step 3: Convert CSV to text docs
    docs = []
    for _, row in df.iterrows():
        text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(text)

    # Step 4: Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.create_documents(docs)

    # Step 5: Embeddings & vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Step 6: Load text generation pipeline
    hf_pipeline = pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",
        token=api_token,
        max_new_tokens=512,
        temperature=0.5,
        do_sample=True,
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Step 7: Build RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # Step 8: Q&A Interface
    st.markdown("---")
    st.subheader("💬 Ask a question about your CSV:")
    question = st.text_input("Type your question:")

    if question:
        with st.spinner("🤖 Generating answer..."):
            answer = qa_chain.run(question)
            st.success(answer)
