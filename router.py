from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)
router_template = """You are a routing assistant for an ecommerce support system.
Classify the user's question into exactly one category:
 
- "sql" - if it asks about order status, order history, stock quantity, prices,
          customer account details, or anything requiring a database lookup
- "vector" - if it asks about product descriptions, return/shipping/warranty policies,
             FAQs, or general information that would be found in documents
- "both" - if it requires BOTH a database lookup (order/customer/stock specifics)
           AND document knowledge (policy/FAQ/product info) to answer correctly

Examples:
Question: "Where is my order #12?"
Category: sql

Question: "Does the jacket run true to size?"
Category: vector

Question: "I ordered a jacket last week, can I still return it for a full refund?"
Category: both

Now classify this question.
Question: {question}

Respond with only one word: sql, vector, or both."""
router_prompt = ChatPromptTemplate.from_template(router_template)
router_chain = router_prompt | llm | StrOutputParser()
def route_question(question: str) -> str:
    """Returns 'sql', 'vector', or 'both'."""
    result = router_chain.invoke({"question": question})
    # clean up in case the model adds extra whitespace/punctuation
    result = result.strip().lower().replace(".", "")
    return result


if __name__ == "__main__":
    test_questions = [
        "Where is my order #3?",
        "Does the jacket run true to size?",
        "I ordered a jacket last week, can I still return it for a full refund?",
        "How many AeroBook laptops do you have in stock?",
        "What payment methods do you accept?",
    ]

    for q in test_questions:
        category = route_question(q)
        print(f"Q: {q}")
        print(f"Route: {category}\n")



