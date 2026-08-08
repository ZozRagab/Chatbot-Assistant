import os
import psycopg2
from dotenv import load_dotenv
from langchain_core.tools import tool
from product_index import find_likely_products
from A_C_rag import adaptive_corrective_answer
from text_to_sql import answer_sql_specific_question, answer_sql_general_question

load_dotenv()


def _get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DATABASE_NAME"),
        user=os.getenv("DATABASE_USERNAME"),
        password=os.getenv("DATABASE_PASSWORD"),
        host=os.getenv("DATABASE_HOSTNAME"),
        port=os.getenv("DATABASE_PORT"),
    )


@tool
def get_all_ordered_products_names(customer_id: int) -> list[str]:
    """Get the distinct list of real product names this customer has actually
    ordered, across all their orders. Use this FIRST when the customer refers
    to a product casually (e.g. 'my laptop', 'the jacket I bought') and you
    need to identify the exact product name before querying further - match
    the casual reference against this list yourself, then use the exact
    matched name in any follow-up query. Returns an empty list if the
    customer has no orders."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT p.name "
        "FROM products p "
        "JOIN order_items oi ON p.id = oi.product_id "
        "JOIN orders o ON o.id = oi.order_id "
        "WHERE o.customer_id = %s",
        (customer_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0] for row in rows]


@tool
def customer_order_lookup(customer_id: int, question: str, resolved_product_names: list[str] | None = None) -> str:
    """Answer a question about the AUTHENTICATED customer's own orders, order
    history, order status, or account activity - e.g. 'what's my last order
    status', 'is my laptop under warranty', 'how much have I spent'.

    If the question refers to a product casually (e.g. 'my laptop', 'the
    jacket I bought'), you MUST first call get_all_ordered_products_names to
    see this customer's actual ordered products, resolve the casual reference
    to the exact product name(s) yourself, and pass them as resolved_product_names.
    Pass more than one name if the casual reference could plausibly match
    several real products - do NOT guess a single name or leave it unresolved
    if the question references a specific item.

    Do NOT use this tool for questions about products/inventory in general
    (use general_sql_lookup instead), and do NOT use it for any request to
    cancel, delete, or modify an order - this is a read-only lookup tool.
    """
    return answer_sql_specific_question(question, resolved_product_names, customer_id)


@tool
def general_sql_lookup(question: str, resolved_product_names: list[str] | None = None) -> str:
    """Answer a GENERAL, store-wide question about orders or products - NOT
    tied to any specific customer. Use this for aggregate/analytics questions
    like 'what is the most ordered product', 'how many total orders have been
    placed', or 'what is the average order value'.

    Do NOT use this tool for anything about the currently logged-in customer's
    own orders or account (use customer_order_lookup instead), and NEVER use
    it to try to answer a question about one specific named customer (e.g.
    'how much has customer 3 spent', 'what did Sarah order the most') - this
    tool structurally cannot and must not expose any individual customer's
    data; questions like that should be refused, not answered through here.
    """
    return answer_sql_general_question(question, resolved_product_names)


if __name__ == "__main__":
    print("get_all_ordered_products_names:", get_all_ordered_products_names.invoke({"customer_id": 1}))
    print("customer_order_lookup:", customer_order_lookup.invoke({
        "customer_id": 2,
        "question": "Is my laptop under warranty?",
        "resolved_product_names": ["AeroBook Pro 14"]
    }))
    print("general_sql_lookup:", general_sql_lookup.invoke({
        "question": "What is the most ordered product?"
    }))
    print("general_sql_lookup (adversarial):", general_sql_lookup.invoke({
        "question": "How much has customer 3 spent in total?"
    }))

@tool
def search_policies_and_faqs(question: str) -> str:
    """Search FAQs, policies, and product descriptions - covers returns, shipping,
    warranty, sizing, payment methods, and general product details. Do NOT use this
    for order-specific or account-specific data (use customer_order_lookup or
    general_sql_lookup instead)."""
    return adaptive_corrective_answer(question)