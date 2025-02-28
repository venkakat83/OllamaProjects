import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings_ollama = OllamaEmbeddings(model="gemma2:2b")
chromaDB = Chroma(embedding_function=embeddings_ollama, 
                collection_name="chromadb_ollama", 
                persist_directory="../chromadb_ollama")

PDF_STORAGE_PATH = '12-02-2025-Bits_Project/RAG_Training/document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="gemma2:2b")

def read_pdfs_from_directory(directory):
    directory = directory
    pdf_texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
                raw_docs = load_pdf_documents(directory + filename)
                processed_chunks = chunk_documents(raw_docs)
                index_documents(processed_chunks)
    return pdf_texts


def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    chromaDB.add_documents(document_chunks)

read_pdfs_from_directory(PDF_STORAGE_PATH)

# Having Filter is important to look for the exact document
# If you don't have a filter, it will look for the similar content in the whole database
results = chromaDB.similarity_search(
    "Detailed Summary of M9F11206",
    k=2,
    filter={"source": "document_store/pdfs/M9F11206.pdf"},
)

for res in results:
    print(f"* {res.page_content} [{res.metadata}]")