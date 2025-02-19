import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="gemma2:2b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="gemma2:2b")
directory = ""

def read_pdfs_from_directory(directory):
    directory = directory
    pdf_texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
                raw_docs = load_pdf_documents(directory + filename)
                processed_chunks = chunk_documents(raw_docs)
                index_documents(processed_chunks)
                # embeddings = convert_to_embeddings(processed_chunks)
                # print(embeddings)
                # save_embeddings_to_chroma(embeddings, filename)
    return pdf_texts

def save_uploaded_file(uploaded_file):
    # Add present working directory
    print("Current working directory:", os.getcwd())
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_file(file_name):
    print(PDF_STORAGE_PATH + file_name + '.pdf')
    raw_docs = load_pdf_documents(file_path= PDF_STORAGE_PATH + file_name + '.pdf')
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)

def load_pdf_documents(file_path):
    is_successful = DOCUMENT_VECTOR_DB.delete()
    print(is_successful)
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def convert_to_embeddings(chunks):
    embedding_model = OllamaEmbeddings(model="gemma2:2b")
    embeddings = [embedding_model.embed_documents(chunk.page_content) for chunk in chunks]
    return embeddings

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    print(context_text)
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})



# query = "Routine operating checks of residual current devices"
# relevant_docs = find_related_documents(query)
# ai_response = generate_answer(query, relevant_docs)
# print(ai_response)

# results = similarity_search(query)
# print("Similarity Search Results:", results)