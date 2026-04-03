import os
import chainlit as cl
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from src.retriever import build_retriever
from src.pipeline import run_gen_val_pipeline

load_dotenv()

GROQ_API_KEY1   = os.environ.get('GROQ_API_KEY1')
GROQ_API_KEY2   = os.environ.get('GROQ_API_KEY2')
HF_TOKEN       = os.environ.get('HF_TOKEN')
HF_REPO        = "your-username/symptomate-index"

generator_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY1
)

validator_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY2
)

_retriever = None


@cl.on_chat_start
async def start():
    global _retriever

    if _retriever is None:
        loading = await cl.Message(content="⏳ Loading documents, please wait...").send()
        _retriever = await cl.make_async(build_retriever)(HF_REPO, HF_TOKEN)
        await loading.remove()

    cl.user_session.set("retriever", _retriever)

    await cl.Message(
        content="👋 Hi! I'm Symptomate. Describe your symptoms and I'll help you.",
        author="Symptomate"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    retriever = cl.user_session.get("retriever")

    msg = cl.Message(content="")
    await msg.send()

    await run_gen_val_pipeline(
        question=message.content,
        retriever=retriever,
        generator_llm=generator_llm,
        validator_llm=validator_llm,
        msg=msg
    )

    await msg.update()