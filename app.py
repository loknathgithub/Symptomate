import os
import pickle
import chainlit as cl
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import hf_hub_download, upload_file
from dotenv import load_dotenv
from src.prompt import *

load_dotenv()

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
HF_TOKEN       = os.environ.get('HF_TOKEN')
HF_REPO        = "LoknathSaha2004/symptomate-index"
PDF_DIR        = "data/"
INDEX_CACHE    = "bm25_index.pkl"

_retriever = None  # shared across sessions

# ── helpers ───────────────────────────────────────────────────────────────────

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_retriever():
    # 1. Local cache
    if os.path.exists(INDEX_CACHE):
        print("✅ Loading from local cache...")
        with open(INDEX_CACHE, "rb") as f:
            return pickle.load(f)

    # 2. Hugging Face Hub
    try:
        print("⬇️  Downloading from Hugging Face...")
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename="bm25_index.pkl",
            repo_type="dataset",
            token=HF_TOKEN
        )
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    # 3. Build from PDFs
    print("⏳ Building index from PDFs (first time only)...")
    docs = PyPDFDirectoryLoader(PDF_DIR).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 3

    with open(INDEX_CACHE, "wb") as f:
        pickle.dump(retriever, f)

    try:
        print("⬆️  Uploading to Hugging Face...")
        upload_file(
            path_or_fileobj=INDEX_CACHE,
            path_in_repo="bm25_index.pkl",
            repo_id=HF_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print("✅ Uploaded.")
    except Exception as e:
        print(f"⚠️  Upload failed (index saved locally): {e}")

    return retriever


def build_chain(retriever):
    return (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )


# ── LLM & prompt ─────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


# ── Chainlit ──────────────────────────────────────────────────────────────────

@cl.on_chat_start
async def start():
    global _retriever

    if _retriever is None:
        msg = await cl.Message(content="⏳ Loading documents, please wait...").send()
        _retriever = await cl.make_async(build_retriever)()  # ← non-blocking
        await msg.remove()

    cl.user_session.set("chain", build_chain(_retriever))
    await cl.Message(
        content="👋 Hi! I'm Symptomate. Describe your symptoms and I'll help you."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    msg = cl.Message(content="")
    await msg.send()

    async for chunk in chain.astream(message.content):
        await msg.stream_token(chunk)

    await msg.update()