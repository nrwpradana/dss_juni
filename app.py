import streamlit as st
import pandas as pd
import tempfile
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Streamlit page configuration
st.set_page_config(page_title="CSV Q&A Chatbot", page_icon="📊")
st.header("RAG CSV Q&A Chatbot 💬")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Enter your Hugging Face API token and upload a CSV file to start chatting!"}]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "hf_token" not in st.session_state:
    st.session_state.hf_token = ""

# Input for Hugging Face API token
st.subheader("Hugging Face API Token")
hf_token = st.text_input("Enter your Hugging Face API token:", type="password", value=st.session_state.hf_token)
if hf_token:
    st.session_state.hf_token = hf_token
    st.success("API token saved for this session.")
else:
    st.warning("Please enter a valid Hugging Face API token to proceed.")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file and hf_token:
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load CSV data using LangChain CSVLoader
        loader = CSVLoader(file_path=tmp_file_path)
        documents = loader.load()
        st.write(f"Loaded {len(documents)} documents from CSV.")

        # Check for sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            st.error("The 'sentence-transformers' package is missing. Please install it with 'pip install sentence-transformers'.")
            os.unlink(tmp_file_path)
            st.stop()

        # Initialize embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vector_store = FAISS.from_documents(documents, embeddings)
        st.write("Vector store initialized with FAISS.")

        # Initialize Hugging Face LLM
        try:
            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                huggingfacehub_api_token=hf_token,
                temperature=0.7,
                task="conversational"  # Set task to conversational
            )
            st.write("Using model: mistralai/Mixtral-8x7B-Instruct-v0.1 with task=conversational")
        except Exception as e:
            st.warning(f"Failed to initialize Mixtral model: {str(e)}. Falling back to google/flan-t5-large.")
            llm = HuggingFaceEndpoint(
                repo_id="google/flan-t5-large",
                huggingfacehub_api_token=hf_token,
                temperature=0.7
            )
            st.write("Using fallback model: google/flan-t5-large")

        # Set up conversation chain with memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vector_store.as_retriever(),
            memory=memory
        )

        # Clean up temporary file
        os.unlink(tmp_file_path)
        st.success("CSV file processed! You can now ask questions.")
    except Exception as e:
        st.error(f"Error processing CSV or initializing model: {str(e)}")
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
elif uploaded_file and not hf_token:
    st.error("Please enter a Hugging Face API token before uploading a CSV.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your CSV data"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response if conversation chain is initialized
    if st.session_state.conversation:
        try:
            with st.chat_message("assistant"):
                response = st.session_state.conversation({"question": prompt})["answer"]
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            with st.chat_message("assistant"):
                st.write(f"Error generating response: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error generating response: {str(e)}"})
    else:
        with st.chat_message("assistant"):
            st.write("Please upload a CSV file and ensure a valid API token is provided.")
            st.session_state.messages.append({"role": "assistant", "content": "Please upload a CSV file and ensure a valid API token is provided."})