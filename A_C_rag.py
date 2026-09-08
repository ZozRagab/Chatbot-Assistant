# A_C_rag.py
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_groq import ChatGroq
from pipeline import (
    fusion_retrieval_chain,
    generation_chain,
    simple_answer_question,
)

load_dotenv()

# Stronger model - used for classification (routing decision matters)
llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}}
)

# Fast model for cheap steps: grading and rewriting
fast_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# ============================================
# STEP 1: Adaptive routing - classify the retrieval strategy needed
# ============================================
adaptive_template = """Classify a customer question as "simple" or "careful".

simple: one direct fact lookup. Shipping, delivery, payment methods, product info, account basics, order status.
careful: refund/return/warranty/voucher rules, eligibility, edge cases, or conditional policy where a wrong answer could mislead the customer.

Examples:
Q: How long does shipping take? -> simple
Q: What is the shipping policy? -> simple
Q: What payment methods do you accept? -> simple
Q: How do I reset my password? -> simple
Q: If I return a gift, do I get cash or store credit? -> careful
Q: Can I return a damaged item after 30 days? -> careful
Q: Do vouchers stack with sale prices? -> careful

Q: {question}
Answer with one word only: simple or careful."""

adaptive_prompt = ChatPromptTemplate.from_template(adaptive_template)
adaptive_chain = adaptive_prompt | fast_llm | StrOutputParser()


def classify_strategy(question: str) -> str:
    result = adaptive_chain.invoke({"question": question})
    return result.strip().lower().replace(".", "")


# ============================================
# STEP 2: Corrective RAG - grade all retrieved chunks in a single batched call
# ============================================
grade_template = """Grade whether each document could help answer the question.

Question: {question}

Documents:
{numbered_documents}

Reply with one line per document in this exact format:
1. yes
2. no
3. yes

No other text."""

grade_prompt = ChatPromptTemplate.from_template(grade_template)
grade_chain = grade_prompt | fast_llm | StrOutputParser()


def grade_chunks(question: str, chunks: list) -> list:
    """Grades all chunks in a single LLM call instead of one call per chunk."""
    if not chunks:
        return []

    numbered_documents = "\n\n".join(
        f"[{i+1}] {c.page_content}" for i, c in enumerate(chunks)
    )
    result = grade_chain.invoke({"question": question, "numbered_documents": numbered_documents})
    print(f"  [RAW batch grade output: {result!r}]")

    verdicts = {}
    for line in result.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.replace(")", ".").split(".", 1)
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[0].strip())
        except ValueError:
            continue
        verdicts[idx] = parts[1].strip().lower().startswith("yes")

    if len(verdicts) != len(chunks):
        print(f"  [WARNING: expected {len(chunks)} verdicts, parsed {len(verdicts)} - check RAW output above]")

    relevant_chunks = [c for i, c in enumerate(chunks) if verdicts.get(i + 1, False)]

    for i, c in enumerate(chunks):
        preview = c.page_content[:60].replace("\n", " ")
        is_relevant = verdicts.get(i + 1, False)
        print(f"    - '{preview}...' -> {'RELEVANT' if is_relevant else 'NOT relevant'}")

    return relevant_chunks


# ============================================
# STEP 3: Corrective RAG - rewrite the query if retrieval graded poorly
# ============================================
rewrite_template = """Rewrite this question to be clearer and more specific for document retrieval.

Original: {question}
Rewritten:"""

rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
rewrite_chain = rewrite_prompt | fast_llm | StrOutputParser()


def rewrite_query(question: str) -> str:
    return rewrite_chain.invoke({"question": question}).strip()


# ============================================
# Main orchestration function
# ============================================
def adaptive_corrective_answer(question: str) -> str:
    strategy = classify_strategy(question)
    print(f"  [CLASSIFIED: {strategy}]")

    if strategy == "simple":
        # Direct retrieval + generation, no fusion, no grading. 1 LLM call.
        return simple_answer_question(question)

    # "careful" - fusion retrieval + batch grade + (optional) rewrite + generation
    retrieved_chunks = fusion_retrieval_chain.invoke({"question": question})
    top_chunks = retrieved_chunks[:5]

    print("  [Grading initial retrieval...]")
    relevant_chunks = grade_chunks(question, top_chunks)

    if not relevant_chunks:
        print("  [No relevant chunks found - rewriting query and retrying...]")
        new_question = rewrite_query(question)
        print(f"  [Rewritten query: '{new_question}']")
        retrieved_chunks = fusion_retrieval_chain.invoke({"question": new_question})
        top_chunks = retrieved_chunks[:5]
        print("  [Grading retry retrieval...]")
        relevant_chunks = grade_chunks(question, top_chunks)

    if not relevant_chunks:
        return "I don't have enough reliable information to answer that confidently."

    context_text = "\n\n".join(chunk.page_content for chunk in relevant_chunks)
    return generation_chain.invoke({"context": context_text, "question": question})


if __name__ == "__main__":
    test_questions = [
        "What is the return policy?",
        "How long does shipping take?",
        "What is the shipping policy?",
        "If I return a gift, do I get cash back or store credit?",
        "If a delivered item is spoiled, do I get my money back automatically, or do I have to request it?",
        "What payment methods do you accept?",
        "Can I combine two voucher codes on one order?",
        "Do you sell umbrellas?"
    ]

    for q in test_questions:
        strategy = classify_strategy(q)
        print(f"\nQ: {q}")
        print(f"Strategy: {strategy}")
        print(f"A: {adaptive_corrective_answer(q)}")