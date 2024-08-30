# import streamlit as st
# import os
# import json
# import time
# from langchain_groq import ChatGroq
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from dotenv import load_dotenv
# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import json

# # Load environment variables
# load_dotenv()

# # Set your API keys
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# # Initialize the GROQ API model
# groq_api_key = os.getenv('GROQ_API_KEY')
# llm = ChatGroq(
#     groq_api_key=groq_api_key,
#     model_name="llama3-8b-8192"
# )

# # Set up the embedding model
# huggingface_embeddings = HuggingFaceBgeEmbeddings(
#     model_name="BAAI/bge-small-en-v1.5",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )

# # Define the prompt template
# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question
#     <context>
#     {context}
#     <context>
#     Questions:{input}
#     """
# )

# # Convert Document objects to dictionaries for JSON serialization
# def document_to_dict(doc):
#     return {
#         "page_content": doc.page_content,
#         "metadata": doc.metadata
#     }

# # Save the response to a file
# def save_response(response):
#     # Convert any Document objects in the response to dictionaries
#     if "context" in response:
#         response["context"] = [document_to_dict(doc) for doc in response["context"]]

#     with open("response.json", "w") as f:
#         json.dump(response, f)

# # Define the vector embedding function
# def vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.huggingface_embeddings = huggingface_embeddings
#         st.session_state.loader = PyPDFDirectoryLoader("../datas")  # Data Ingestion
#         st.session_state.docs = st.session_state.loader.load()  # Document Loading
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.huggingface_embeddings)  # Vector embeddings

# # Main Streamlit code
# st.title("Aahar")

# prompt1 = st.text_input("Enter Your Question From Documents")

# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB Is Ready")

# if prompt1:
#     if "vectors" in st.session_state:
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)
#         start = time.process_time()
#         response = retrieval_chain.invoke({'input': prompt1})
#         print("Response time:", time.process_time() - start)
#         st.write(response['answer'])

#         # Save response to file
#         save_response(response)

#         with st.expander("Document Similarity Search"):
#             for i, doc in enumerate(response["context"]):
#                 st.write(doc["page_content"])
#                 st.write("--------------------------------")
#     else:
#         st.write("Please embed the documents first by clicking the 'Documents Embedding' button.")


# # FastAPI Setup
# app = FastAPI()

# JSON_FILE_PATH = "question.json"

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Allow your React app's origin
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
#     allow_headers=["*"],  # Allow all headers
# )

# def save_question(question):
#     try:
#         with open(JSON_FILE_PATH, "r+") as f:
#             data = json.load(f)
#             data.append(question)
#             f.seek(0)
#             json.dump(data, f, indent=4)
#     except FileNotFoundError:
#         with open(JSON_FILE_PATH, "w") as f:
#             json.dump([question], f, indent=4)

# async def process_questions():
#     try:
#         with open(JSON_FILE_PATH, "r") as f:
#             questions = json.load(f)
        
#         if questions:
#             # Assuming the last question needs to be processed
#             last_question = questions[-1]["question"]

#             # Use the chatbot logic to process the question
#             response = process_question_with_bot(last_question)

#             # Save the response to response.json
#             save_response(response)
#     except Exception as e:
#         print(f"Error processing questions: {str(e)}")

# def process_question_with_bot(question):
#     if "vectors" in st.session_state:
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)
#         response = retrieval_chain.invoke({'input': question})
#         return response['answer']
#     else:
#         raise ValueError("Document embeddings are not initialized.")

# @app.post("/post-question")
# async def post_question(request: Request):
#     try:
#         question = await request.json()
#         save_question(question)

        

#         return JSONResponse(content={"message": "Question saved successfully."})
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# @app.get("/get-response")
# async def get_response():
#     try:
#         with open("response.json", "r") as f:
#             response = json.load(f)
#         return JSONResponse(content=response)
#     except FileNotFoundError:
#         return JSONResponse(content={"error": "No response found."}, status_code=404)
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)




# import streamlit as st
# import os
# import json
# import time
# from langchain_groq import ChatGroq
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from dotenv import load_dotenv
# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware

# # Load environment variables
# load_dotenv()

# # Set your API keys
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# # Initialize the GROQ API model
# groq_api_key = os.getenv('GROQ_API_KEY')
# llm = ChatGroq(
#     groq_api_key=groq_api_key,
#     model_name="llama3-8b-8192"
# )

# # Set up the embedding model
# huggingface_embeddings = HuggingFaceBgeEmbeddings(
#     model_name="BAAI/bge-small-en-v1.5",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )

# # Define the prompt template
# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question.
#     <context>
#     {context}
#     <context>
#     Questions: {input}
#     """
# )

# # Convert Document objects to dictionaries for JSON serialization
# def document_to_dict(doc):
#     return {
#         "page_content": doc.page_content,
#         "metadata": doc.metadata
#     }

# # Save the response to a file
# def save_response(response):
#     # Convert any Document objects in the response to dictionaries
#     if "context" in response:
#         response["context"] = [document_to_dict(doc) for doc in response["context"]]

#     with open("response.json", "w") as f:
#         json.dump(response, f, indent=4)

# # Define the vector embedding function
# def vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.huggingface_embeddings = huggingface_embeddings
#         st.session_state.loader = PyPDFDirectoryLoader("../datas")  # Data Ingestion
#         st.session_state.docs = st.session_state.loader.load()  # Document Loading
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.huggingface_embeddings)  # Vector embeddings

# # Load the latest question from the JSON file
# def get_latest_question():
#     try:
#         with open("question.json", "r") as f:
#             questions = json.load(f)
#         return questions[-1]["text"] if questions else ""
#     except FileNotFoundError:
#         return ""

# # Main Streamlit code
# st.title("Aahar")

# # Automatically fetch the latest question every few seconds
# latest_question = get_latest_question()

# # Display the latest question in the text input box
# prompt1 = st.text_input("Enter Your Question From Documents", value=latest_question)

# # if prompt1 and prompt1 == latest_question:
# #     st.experimental_rerun() 
# def trigger_check_for_new_questions():
#     st.experimental_rerun()
# # Button to check for new questions manually
# if st.button("Check for New Questions"):
#     trigger_check_for_new_questions()

# # Button to initialize vector embeddings
# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB Is Ready")

# # If a question is present, process it
# if prompt1:
#     if "vectors" in st.session_state:
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)
#         start = time.process_time()
#         response = retrieval_chain.invoke({'input': prompt1})
#         print("Response time:", time.process_time() - start)
#         st.write(response['answer'])

#         # Save the response to file
#         save_response(response)

#         # Display document similarity search results
#         with st.expander("Document Similarity Search"):
#             for i, doc in enumerate(response["context"]):
#                 st.write(doc["page_content"])
#                 st.write("--------------------------------")
#     else:
#         st.write("Please embed the documents first by clicking the 'Documents Embedding' button.")

# # FastAPI Setup
# app = FastAPI()

# # Enable CORS for the React app running on localhost:3000
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Allow your React app's origin
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
#     allow_headers=["*"],  # Allow all headers
# )

# # Save the question to question.json
# def save_question(question):
#     try:
#         with open("question.json", "r+") as f:
#             data = json.load(f)
#             data.append(question)
#             f.seek(0)
#             json.dump(data, f, indent=4)
#     except FileNotFoundError:
#         with open("question.json", "w") as f:
#             json.dump([question], f, indent=4)

# # FastAPI endpoint to receive a question from the website
# @app.post("/post-question")
# async def post_question(request: Request):
#     try:
#         question = await request.json()
        
#         save_question(question)  # Save the question to question.json

#         # Optional: Process the question immediately
#         # await process_questions()
        
#         return JSONResponse(content={"message": "Question saved and processed successfully."})
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# # FastAPI endpoint to get the response
# @app.get("/get-response")
# async def get_response():
#     try:
#         with open("response.json", "r") as f:
#             response = json.load(f)
#         return JSONResponse(content=response)
#     except FileNotFoundError:
#         return JSONResponse(content={"error": "No response found."}, status_code=404)
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

import streamlit as st
import os
import json
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

st.markdown(
    """
    <style>
    /* Background color for the main content area */
    # .stApp {
    #     background-color: #FFEFD5;  /* Light gray */
    # }
    .stTitle {
        color: #4B0082;  /* Indigo */
        font-size: 2.5em;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #005f6b; /* Darker blue */
        color: #fff;
    }

    /* Background color for the sidebar */
    .css-1d391kg {  
        background-color: #4CAF50 !important;  /* Green */
    }

    /* Additional customization (optional) */
    .css-18e3th9 {
        background-color: #4CAF50 !important;  /* Green sidebar elements */
    }
    .stButton button {
        background-color: #008CBA; /* Blue */
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stTextInput > div > div > input {
        background-color: #FFF8DC; /* Cornsilk */
        color: #000;
        border: 2px solid #000;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Load environment variables
load_dotenv()

# Set your API keys
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# Initialize the GROQ API model
groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"
)

# Set up the embedding model
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Convert Document objects to dictionaries for JSON serialization
def document_to_dict(doc):
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }

# Save the response to a file
def save_response(response):
    # Convert any Document objects in the response to dictionaries
    if "context" in response:
        response["context"] = [document_to_dict(doc) for doc in response["context"]]

    with open("response.json", "w") as f:
        json.dump(response, f, indent=4)

# Define the vector embedding function
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.huggingface_embeddings = huggingface_embeddings
        st.session_state.loader = PyPDFDirectoryLoader("../datas")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.huggingface_embeddings)  # Vector embeddings

# Load the latest question from the JSON file
def get_latest_question():
    try:
        with open("question.json", "r") as f:
            questions = json.load(f)
        return questions[-1]["text"] if questions else ""
    except FileNotFoundError:
        return ""

# Main Streamlit code
# st.title("Aahar")
st.image("../Aahar.png", width=300)

# Automatically fetch the latest question every few seconds
latest_question = get_latest_question()
a_prev_response = latest_question

# Display the latest question in the text input box
prompt1 = st.text_input("Enter Your Question From Documents", value=latest_question)

# if prompt1 and prompt1 == latest_question:
#     st.experimental_rerun() 
def trigger_check_for_new_questions():
    st.experimental_rerun()

# Button to check for new questions manually
if st.button("Submit"):
    trigger_check_for_new_questions()

# Button to initialize vector embeddings
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

# If a question is present, process it
if prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # Save the response to file
        save_response(response)

        # Display document similarity search results
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc["page_content"])
                st.write("--------------------------------")
    else:
        st.write("Please embed the documents first by clicking the 'Documents Embedding' button.")

# FastAPI Setup
app = FastAPI()

# Enable CORS for the React app running on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your React app's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Save the question to question.json
def save_question(question):
    try:
        with open("question.json", "r+") as f:
            data = json.load(f)
            data.append(question)
            f.seek(0)
            json.dump(data, f, indent=4)
    except FileNotFoundError:
        with open("question.json", "w") as f:
            json.dump([question], f, indent=4)

# FastAPI endpoint to receive a question from the website
@app.post("/post-question")
async def post_question(request: Request):
    try:
        question = await request.json()
        
        save_question(question)  # Save the question to question.json

        # Optional: Process the question immediately
        # await process_questions()
        
        return JSONResponse(content={"message": "Question saved and processed successfully."})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# FastAPI endpoint to get the response
@app.get("/get-response")
async def get_response():
    try:
        with open("response.json", "r") as f:
            response = json.load(f)
        return JSONResponse(content=response)
    except FileNotFoundError:
        return JSONResponse(content={"error": "No response found."}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
