import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Determine paths relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(CURRENT_DIR, "docs")
VECTORSTORE_DIR = os.path.join(CURRENT_DIR, "vectorstore")

def build_education_vectorstore():
    if not os.path.exists(DOCS_DIR):
        raise FileNotFoundError(f"Directory not found: {DOCS_DIR}. Ensure your PDFs are inside backend/rag/docs/")

    start_time = time.time()
    print("=" * 60)
    print("STEP 1: Loading PDF policy documents from 'backend/rag/docs/'...")
    print("=" * 60)
    
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    documents = loader.load()
    print(f"--> Successfully loaded {len(documents)} pages across all policy documents.")

    print("\n" + "=" * 60)
    print("STEP 2: Splitting text into semantic chunks (1000 chars, 200 overlap)...")
    print("=" * 60)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"--> Created {len(chunks)} searchable chunks from {len(documents)} pages.")

    print("\n" + "=" * 60)
    print("STEP 3: Generating local vector embeddings...")
    print("        Model: sentence-transformers/all-MiniLM-L6-v2 (Runs locally on CPU)")
    print("=" * 60)
    
    # Download/load lightweight local HuggingFace embedding model (384 dimensions, zero API cost)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("\n" + "=" * 60)
    print("STEP 4: Writing vectors into FAISS index...")
    print("=" * 60)
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    
    elapsed = round(time.time() - start_time, 2)
    print(f"\n[SUCCESS] Vector database created and saved to: '{VECTORSTORE_DIR}'")
    print(f"Total time taken: {elapsed} seconds.")
    print("=" * 60)

if __name__ == "__main__":
    build_education_vectorstore()