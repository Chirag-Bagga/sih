import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

load_dotenv()

# Load the GROQ API KEY
groq_api_key = os.getenv('GROQ_API_KEY')

# Streamlit Title
st.title("Chatgroq With Llama3 Demo")

# Initialize session state for embeddings and vectors
if "vectors" not in st.session_state:
    st.session_state.embedding = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embedding)

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name='Gemma-7b-it')

# Corrected Prompt Template
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Create the document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# User input prompt
user_prompt = st.text_input("Input your prompt here")

# If a prompt is provided, execute the retrieval chain
if user_prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    response_time = time.process_time() - start
    
    st.write(f"Response time: {response_time:.2f} seconds")
    st.write(response['answer'])

    # Display document similarity search results
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

# Debugging: Show which model is being used
st.write(f"Using model: {llm.model_name}")
