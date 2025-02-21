from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings_ollama = OllamaEmbeddings(model="gemma2:2b")
chromaDB = Chroma(embedding_function=embeddings_ollama, 
                collection_name="chromadb_ollama", 
                persist_directory="./chromadb_ollama")


results = chromaDB.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"},
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")