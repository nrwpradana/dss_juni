import streamlit as st
import pandas as pd
import os
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Setup page
st.set_page_config(page_title="CSV Q&A Chatbot", layout="centered")
st.title("📊 CSV Q&A Chatbot (Fast & Light via flan-t5-base)")

# Step 1: API token input (secure)
api_token = st.text_input("🔐 Enter your Hugging Face API token:", type="password")
if not api_token:
    st.info("Please enter your Hugging Face API token to continue.")
    st.stop()

# Step 2: Upload CSV
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])
if not uploaded_file:
    st.stop()

# Step 3: Read and preview CSV
df = pd.read_csv(uploaded_file)
st.subheader("📄 CSV Preview")
st.dataframe(df.head())

# Step 4: Convert rows to documents
docs = []
for _, row in df.iterrows():
    text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
    docs.append(text)

# Step 5: Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.create_documents(docs)

# Step 6: Create embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# Step 7: Setup Hugging Face pipeline with flan-t5-base (FAST)
flan_pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    token=api_token,
    max_new_tokens=256,
    temperature=0.3,
)

llm = HuggingFacePipeline(pipeline=flan_pipe)

# Step 8: Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Step 9: Q&A interface
st.markdown("---")
st.subheader("💬 Ask a question about your CSV")

question = st.text_input("Type your question:")
if question:
    with st.spinner("🤖 Generating answer..."):
        answer = qa_chain.run(question)
        st.success(answer)
