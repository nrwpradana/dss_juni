import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import uuid

# Hugging Face API Key input
hf_api_key = st.sidebar.text_input("🔑 Enter Hugging Face API Key", type="password")

# Streamlit UI
st.title("📊 CSV Q&A Chatbot")
st.write("Upload a CSV and ask questions based on its content!")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload a CSV", type="csv")
if not hf_api_key:
    st.warning("Please enter your Hugging Face API key to proceed.")
    st.stop()
else:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key

# Initialize session state for unique user isolation
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Unique ID for each user session

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = None

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from CSV
def extract_text_from_csv(csv_path):
    # Read CSV into a pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Convert each row to a text representation
    text_chunks = []
    for index, row in df.iterrows():
        # Combine all columns into a single string for the row
        row_text = " | ".join([f"{col}: {str(val)}" for col, val in row.items()])
        text_chunks.append(row_text)
    
    # Combine all rows into a single text string
    all_text = "\n\n".join(text_chunks)
    return all_text, text_chunks

# Function to store text embeddings in FAISS
def create_faiss_index(text_data):
    text_chunks = text_data.split("\n\n")  # Split text into chunks (one per row)
    embeddings = np.array([embedding_model.encode(chunk) for chunk in text_chunks])

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    return index, text_chunks

# Function to retrieve relevant text
def retrieve_relevant_text(query, index, text_chunks, top_k=3):
    query_embedding = np.array([embedding_model.encode(query)])
    distances, indices = index.search(query_embedding, top_k)
    
    return [text_chunks[i] for i in indices[0]]

# Load LLM
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3", model_kwargs={"temperature": 0.5})

# Define prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Answer based on the context below:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)

qa_chain = LLMChain(llm=llm, prompt=prompt)

# Process uploaded CSV
if uploaded_file and hf_api_key:
    st.write("✅ CSV uploaded successfully!")

    # Create a unique file path for the user's session
    csv_path = f"uploaded_{st.session_state.session_id}.csv"
    
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract content from CSV
    all_text_data, text_chunks = extract_text_from_csv(csv_path)

    # Create FAISS index (store it per user session)
    index, text_chunks = create_faiss_index(all_text_data)
    
    st.session_state.faiss_index = index
    st.session_state.text_chunks = text_chunks

    st.success("✅ CSV processed! You can now ask questions.")

# User input for Q&A
user_question = st.text_input("Ask a question about the CSV:")
if user_question and st.session_state.faiss_index:
    relevant_chunks = retrieve_relevant_text(user_question, st.session_state.faiss_index, st.session_state.text_chunks)
    context = "\n".join(relevant_chunks)

    # Generate answer
    answer = qa_chain.run({"context": context, "question": user_question})

    st.write("### Answer:")
    st.write(answer)
else:
    st.warning("Please upload a CSV and enter a Hugging Face API Key.")