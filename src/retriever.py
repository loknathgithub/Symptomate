import os
import pickle
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from huggingface_hub import hf_hub_download, upload_file

PDF_DIR     = "data/"
INDEX_CACHE = "bm25_index.pkl"


def build_retriever(hf_repo: str, hf_token: str) -> BM25Retriever:
    print("inside: build retriever")
    # 1. Local cache
    if os.path.exists(INDEX_CACHE):
        print("✅ Loading from local cache...")
        with open(INDEX_CACHE, "rb") as f:
            return pickle.load(f)

    # 2. Hugging Face Hub
    try:
        print("⬇️  Downloading from Hugging Face...")
        path = hf_hub_download(
            repo_id=hf_repo,
            filename="bm25_index.pkl",
            repo_type="dataset",
            token=hf_token
        )
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    # 3. Build from PDFs
    print("⏳ Building index from PDFs (first time only)...")
    docs = PyPDFDirectoryLoader(PDF_DIR).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    ).split_documents(docs)

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 9

    # Save locally
    with open(INDEX_CACHE, "wb") as f:
        pickle.dump(retriever, f)

    # Upload to HF
    try:
        print("⬆️  Uploading to Hugging Face...")
        upload_file(
            path_or_fileobj=INDEX_CACHE,
            path_in_repo="bm25_index.pkl",
            repo_id=hf_repo,
            repo_type="dataset",
            token=hf_token
        )
        print("✅ Uploaded.")
    except Exception as e:
        print(f"⚠️  Upload failed (saved locally): {e}")

    return retriever


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)