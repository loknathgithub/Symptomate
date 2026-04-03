import chainlit as cl
from langchain_core.output_parsers import StrOutputParser
from src.prompts import (
    generate_prompt,
    validate_prompt,
    regenerate_prompt,
    final_validate_prompt
)
from src.retriever import format_docs


async def run_gen_val_pipeline(
    question: str,
    retriever,
    generator_llm,
    validator_llm,
    msg: cl.Message
):
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # ── Round 1: Generate (hidden) ───────────────────────────────────────────
    async with cl.Step(name="⚙️ Generation 1", show_input=False) as step1:
        answer1 = ""
        async for chunk in (generate_prompt | generator_llm | StrOutputParser()).astream(
            {"input": question, "context": context}
        ):
            answer1 += chunk
        step1.output = answer1

    # ── Round 1: Validate (hidden) ───────────────────────────────────────────
    async with cl.Step(name="🔍 Validation 1", show_input=False) as step2:
        validation1 = ""
        async for chunk in (validate_prompt | validator_llm | StrOutputParser()).astream(
            {"question": question, "context": context, "answer": answer1}
        ):
            validation1 += chunk
        step2.output = validation1

    # Extract critique
    critique = next(
        (l.replace("CRITIQUE:", "").strip()
         for l in validation1.splitlines()
         if l.startswith("CRITIQUE:")),
        "No critique provided."
    )

    # ── Round 2: Regenerate (hidden) ─────────────────────────────────────────
    async with cl.Step(name="⚙️ Generation 2 — Improved", show_input=False) as step3:
        answer2 = ""
        async for chunk in (regenerate_prompt | generator_llm | StrOutputParser()).astream(
            {"input": question, "context": context,
             "previous_answer": answer1, "critique": critique}
        ):
            answer2 += chunk
        step3.output = answer2

    # ── Round 2: Final Validate (hidden) ─────────────────────────────────────
    async with cl.Step(name="🔍 Final Validation", show_input=False) as step4:
        validation2 = ""
        async for chunk in (final_validate_prompt | validator_llm | StrOutputParser()).astream(
            {"question": question, "context": context, "answer": answer2}
        ):
            validation2 += chunk
        step4.output = validation2

    # ── Stream only the final answer to the user ─────────────────────────────
    for chunk in answer2.split(" "):
        await msg.stream_token(chunk + " ")