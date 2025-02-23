import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import os
from langchain_chroma import Chroma

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

embeddings_ollama = OllamaEmbeddings(model="gemma2:2b")
LANGUAGE_MODEL = OllamaLLM(model="gemma2:2b")

chromaDB = Chroma(embedding_function=embeddings_ollama, 
                collection_name="chromadb_ollama", 
                persist_directory="./chromadb_ollama")

def find_related_documents(query):
    return chromaDB.similarity_search(query, filter={"source": selected_pdf}, k=2)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})


# UI Configuration


st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

context = st.chat_input("Enter Context")
selected_pdf = ""
if context:
    selected_pdf = "document_store/pdfs/"+ context +".pdf"
    st.write("Selected Context is : " + context)
    st.write(selected_pdf)
    
    
    
user_input = st.chat_input("Enter your question about the document...")
    
if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.spinner("Analyzing document..."):
        relevant_docs = find_related_documents(user_input)
        ai_response = generate_answer(user_input, relevant_docs)
        
    with st.chat_message("assistant", avatar="🤖"):
        st.write(ai_response)