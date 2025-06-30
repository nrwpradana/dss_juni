# Importing Libraries
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceHubEmbeddings

# Extracting text from CSV file
def get_text_from_csv(csv_file) -> str:
    # Read the CSV file
    df = pd.read_csv(csv_file)
    # Assuming the CSV has a column named 'content' with the text data
    # Modify the column name if your CSV uses a different one
    if 'content' not in df.columns:
        raise ValueError("CSV must contain a 'content' column with text data")
    text = " ".join(df['content'].astype(str).tolist())
    return text

# Processing & Converting the texts into chunks
def get_chunks(text: str) -> list:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Getting Vector Database and Embeddings
def get_vectors(chunks: list):
    embeddings = HuggingFaceHubEmbeddings(
        model="avsolatorio/GIST-Embedding-v0",
        huggingfacehub_api_token="hf_token_here"
    )
    vectors = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectors

# Getting Conversational Chain 
def get_conversational_chain(vectors):
    llm = HuggingFaceHub(
        repo_id='google/flan-t5-large',
        model_kwargs={"temperature": 0.9, "max_length": 2048}
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectors.as_retriever(),
        memory=memory
    )
    return conversational_chain

# Main Function
def main():
    st.set_page_config(
        page_title='RAG Chatbot',
        page_icon="🤖"
    )
    st.title("RAG Chatbot 🤖", anchor=None)
    load_dotenv()

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # File uploader for CSV
    csv_file = st.file_uploader("Upload a CSV", type="csv", label_visibility='hidden')

    if csv_file:
        with st.spinner("Processing the CSV"):
            raw_text = get_text_from_csv(csv_file)
            text_chunks = get_chunks(raw_text)
            vector = get_vectors(text_chunks)
            st.session_state.conversation = get_conversational_chain(vector)

    # Chat input
    question = st.chat_input("Ask anything")
    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        with st.chat_message("user"):
            st.markdown(question)
        with st.spinner("Thinking"):
            response = st.session_state.conversation({'question': question})
            st.session_state.chat_history = response['chat_history']
            st.session_state.chat_history.append({
                "role": "Assistant",
                "content": response['answer']
            })
            with st.chat_message("Assistant"):
                st.markdown(response['answer'])

if __name__ == '__main__':
    main()