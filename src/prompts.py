from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are a medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n"
    "{context}"
)

generate_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

validate_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a medical fact-checker.
You will be given a user question, retrieved context, and a generated answer.
Your job is to:
1. Check if the answer is accurate based on the context
2. Identify any missing points or errors
3. Give a short critique

Respond in this format:
CRITIQUE: <your critique here>
NEEDS_IMPROVEMENT: <YES or NO>"""),
    ("human", """Question: {question}
Context: {context}
Answer: {answer}"""),
])

regenerate_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", """Original question: {input}
Your previous answer: {previous_answer}
Critic's feedback: {critique}

Now provide an improved answer addressing the critique."""),
])

final_validate_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior medical reviewer.
Given the question, context, and final answer, provide:
1. A confidence score (0-100%)
2. One line summary of quality

Respond in this format:
CONFIDENCE: <score>%
SUMMARY: <one line quality summary>"""),
    ("human", """Question: {question}
Context: {context}
Final Answer: {answer}"""),
])