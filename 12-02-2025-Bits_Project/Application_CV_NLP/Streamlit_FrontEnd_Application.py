import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import os
import requests
from Utility import load_pdf_file, read_pdfs_from_directory, find_related_documents, generate_answer
import Utility

st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* Custom Reset Button Styling */
    .stButton > button {
        background-color: #FF4B4B !important;
        color: #FFFFFF !important;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background-color: #FF0000 !important;
    }
    
    /* Custom File Uploader Styling */
    .stFileUploader label {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

PDF_STORAGE_PATH = 'document_store/pdfs/'

st.title("🤖 Schneider Document AI")
st.markdown("Your Intelligent Document Assistant")
st.markdown("---")

# Image Upload Section
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

uploaded_image = st.file_uploader(
    "Upload an Image",
    type=["png", "jpg", "jpeg"],
    help="Select an image for analysis",
    accept_multiple_files=False
)

if uploaded_image:
    st.session_state.uploaded_image = uploaded_image

if st.session_state.uploaded_image:
    # Display the uploaded image
    st.image(st.session_state.uploaded_image, caption="Uploaded Image")
    
    # Send the image to the backend API
    with st.spinner("Sending image to backend..."):
        files = {"image": st.session_state.uploaded_image.getvalue()}
        response = requests.post("http://127.0.0.1:5000/upload", files=files)
        
        if response.status_code == 200:
            st.success("✅ Image processed successfully!")
            st.write("Response from backend:", response.json())
            # print(response.json()["label"])
            load_pdf_file(response.json()["label"])
        else:
            st.error("❌ Failed to process the image. Please try again.")

# Reset button
if st.button("Reset"):
    st.session_state.uploaded_image = None
    st.rerun()


# read_pdfs_from_directory(PDF_STORAGE_PATH) 
user_input = st.chat_input("Enter your question about the document...")
    
if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)

# File Upload Section
# uploaded_pdf = st.file_uploader(
#     "Upload Research Document (PDF)",
#     type="pdf",
#     help="Select a PDF document for analysis",
#     accept_multiple_files=False

# )

# if uploaded_pdf:
#     saved_path = Utility.save_uploaded_file(uploaded_pdf)
#     raw_docs = Utility.load_pdf_documents(saved_path)
#     processed_chunks = Utility.chunk_documents(raw_docs)
#     Utility.index_documents(processed_chunks)
    
#     st.success("✅ Document processed successfully! Ask your questions below.")
    
#     user_input = st.chat_input("Enter your question about the document...")
    
#     if user_input:
#         with st.chat_message("user"):
#             st.write(user_input)
        
#         with st.spinner("Analyzing document..."):
#             relevant_docs = find_related_documents(user_input)
#             ai_response = generate_answer(user_input, relevant_docs)
            
#         with st.chat_message("assistant", avatar="🤖"):
#             st.write(ai_response)

