from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from pipeline import fusion_retrieval_chain, generation_chain, answer_question

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ============================================
# STEP 1: Adaptive routing - classify the retrieval strategy needed
# ============================================
adaptive_template = """Classify what kind of approach this question needs.

- "no_retrieval" - the question is generic chit-chat, a greeting, or something
  answerable without looking anything up (e.g. "hello", "thank you")
- "simple" - a single, straightforward document lookup will answer this well
  (most FAQ-style questions: shipping times, payment methods, basic product info)
- "careful" - the question is about something where getting it exactly right matters,
  is ambiguous, or touches policy details where a wrong/incomplete answer could
  mislead the customer (e.g. returns, refunds, warranty, eligibility questions)

Question: {question}

Respond with only one word: no_retrieval, simple, or careful."""

adaptive_prompt = ChatPromptTemplate.from_template(adaptive_template)
adaptive_chain = adaptive_prompt | llm | StrOutputParser()


def classify_strategy(question: str) -> str:
    result = adaptive_chain.invoke({"question": question})
    return result.strip().lower().replace(".", "")


# ============================================
# STEP 2: Corrective RAG - grade each retrieved chunk's relevance
# ============================================
grade_template = """You are grading whether a retrieved document is relevant to a question.

Question: {question}

Retrieved document:
{document}

Is this document relevant and sufficient to help answer the question?
Respond with only one word: yes or no."""

grade_prompt = ChatPromptTemplate.from_template(grade_template)
grade_chain = grade_prompt | llm | StrOutputParser()


def grade_chunk(question: str, chunk_text: str) -> bool:
    result = grade_chain.invoke({"question": question, "document": chunk_text})
    return result.strip().lower().startswith("yes")


# ============================================
# STEP 3: Corrective RAG - rewrite the query if retrieval graded poorly
# ============================================
rewrite_template = """The original question below did not retrieve good enough results.
Rewrite it to be clearer and more specific, to improve document retrieval.

Original question: {question}

Rewritten question:"""

rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
rewrite_chain = rewrite_prompt | llm | StrOutputParser()


def rewrite_query(question: str) -> str:
    return rewrite_chain.invoke({"question": question}).strip()


# ============================================
# Main orchestration function
# ============================================
def adaptive_corrective_answer(question: str) -> str:
    strategy = classify_strategy(question)

    if strategy == "no_retrieval":
        direct_chain = ChatPromptTemplate.from_template(
            "Respond naturally to this message: {question}"
        ) | llm | StrOutputParser()
        return direct_chain.invoke({"question": question})

    elif strategy == "simple":
        return answer_question(question)

    else:  # "careful" - full grade-and-retry loop
        retrieved_chunks = fusion_retrieval_chain.invoke({"question": question})
        top_chunks = retrieved_chunks[:5]

        # grade each chunk
        relevant_chunks = [c for c in top_chunks if grade_chunk(question, c.page_content)]

        if not relevant_chunks:
            # correction: rewrite the query and retry retrieval ONCE
            new_question = rewrite_query(question)
            retrieved_chunks = fusion_retrieval_chain.invoke({"question": new_question})
            top_chunks = retrieved_chunks[:5]
            relevant_chunks = [c for c in top_chunks if grade_chunk(question, c.page_content)]

        if not relevant_chunks:
            return "I don't have enough reliable information to answer that confidently."

        context_text = "\n\n".join(chunk.page_content for chunk in relevant_chunks)
        return generation_chain.invoke({"context": context_text, "question": question})


if __name__ == "__main__":
    test_questions = [
        "hello!",                                                    # no_retrieval
        "How long does shipping take?",                               # simple
        "If I return a gift, do I get cash back or store credit?",    # careful
    ]

    for q in test_questions:
        strategy = classify_strategy(q)
        print(f"\nQ: {q}")
        print(f"Strategy: {strategy}")
        print(f"A: {adaptive_corrective_answer(q)}")