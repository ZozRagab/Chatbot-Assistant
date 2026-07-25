from router import route_question
from text_to_sql import answer_sql_question
from A_C_rag import adaptive_corrective_answer as answer_vector_question
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ============================================
# Prompt used ONLY for the "both" case - combines SQL + vector results
# into one final, coherent answer
# ============================================
combine_template = """You are answering a customer's question using two sources of information:
1. Database results (specific facts about their order/account)
2. Document knowledge (general policies/product info)

Combine both pieces of information into ONE natural, helpful answer.

Question: {question}

Database information:
{sql_answer}

Document/policy information:
{vector_answer}

Final answer:"""

combine_prompt = ChatPromptTemplate.from_template(combine_template)
combine_chain = combine_prompt | llm | StrOutputParser()


def answer(question: str, customer_id: int | None = None) -> tuple[str, str]:
    """Main entry point. Routes the question and returns (route, final_answer)."""
    category = route_question(question)

    if category == "sql":
        return category, answer_sql_question(question, customer_id)

    elif category == "vector":
        return category, answer_vector_question(question)

    elif category == "both":
        sql_answer = answer_sql_question(question, customer_id)
        vector_answer = answer_vector_question(question)
        final = combine_chain.invoke({
            "question": question,
            "sql_answer": sql_answer,
            "vector_answer": vector_answer
        })
        return category, final

    else:
        # fallback in case the router returns something unexpected
        return category, answer_vector_question(question)


if __name__ == "__main__":
    test_questions = [
        "Where is order 3?",                                            # sql
        "Does the jacket run true to size?",                             # vector
        "I ordered a jacket, can I still return it for a full refund?",  # both
    ]

    for q in test_questions:
        route, response_text = answer(q)
        print(f"\nQ: {q}")
        print(f"Route: {route}")
        print(f"A: {response_text}")