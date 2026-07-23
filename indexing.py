import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader

# Load all documents from the docs folder
docs_folder = "docs"
all_documents = []

for filename in os.listdir(docs_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(docs_folder, filename)
        loader = TextLoader(filepath, encoding="utf-8")
        loaded_doc = loader.load()  # returns a list with one Document object
        all_documents.extend(loaded_doc)

print(f"Loaded {len(all_documents)} documents")

# Split each document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_documents)

print(f"Split into {len(chunks)} chunks")

# Set up the embedding model (local, free)
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Embed chunks and store them in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedder,
    collection_name="ecommerce_docs",
    persist_directory="./chroma_data"
)

print("Indexing complete. ChromaDB collection 'ecommerce_docs' created.")

