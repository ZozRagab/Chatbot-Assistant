import os
import psycopg2
from dotenv import load_dotenv
from langchain_core.tools import tool
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
def get_all_ordered_products_names(user_id: int) -> list[str]:
    """Get the distinct list of real product names this user has actually
    ordered, across all their orders. Use this FIRST when the user refers
    to a product casually (e.g. 'my apples', 'the bread I bought') and you
    need to identify the exact product name before querying further - match
    the casual reference against this list yourself, then use the exact
    matched name in any follow-up query. Returns an empty list if the
    user has no orders."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT DISTINCT "Product"."Name" '
        'FROM "Product" '
        'JOIN "Order_Item" ON "Product"."Id" = "Order_Item"."ProductId" '
        'JOIN "Orders" ON "Orders"."Id" = "Order_Item"."OrderId" '
        'WHERE "Orders"."UserId" = %s',
        (user_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0] for row in rows]


@tool
def user_order_lookup(user_id: int, question: str, resolved_product_names: list[str] | None = None) -> str:
    """Answer a question about the AUTHENTICATED user's own orders, order
    history, order status, cart, saved addresses, or reviews THEY wrote -
    e.g. 'what's my last order status', 'is my order delivered yet',
    'what's in my cart', 'what did I review'.

    If the question refers to a product casually (e.g. 'my apples', 'the
    bread I bought'), you MUST first call get_all_ordered_products_names to
    see this user's actual ordered products, resolve the casual reference
    to the exact product name(s) yourself, and pass them as
    resolved_product_names. Pass more than one name if the casual reference
    could plausibly match several real products.

    Do NOT use this tool for questions about products/catalog/reviews in
    general (use general_sql_lookup or check_stock instead), and do NOT use
    it for any request to cancel, delete, or modify an order - this is a
    read-only lookup tool. NEVER use it to answer aggregate/store-wide
    questions - use general_sql_lookup for those.
    """
    return answer_sql_specific_question(question, resolved_product_names, user_id)


@tool
def general_sql_lookup(question: str, resolved_product_names: list[str] | None = None) -> str:
    """Answer a GENERAL, store-wide question about products, categories,
    reviews (by product, not by person), or vouchers - NOT tied to any
    specific user. Use for questions like 'what's our best-selling product',
    'what's the average rating for X', 'is voucher code SAVE20 still valid'.

    Do NOT use this for anything about the currently logged-in user's own
    orders/cart/account (use user_order_lookup instead), and NEVER use it to
    answer a question about one specific named person's reviews or voucher
    usage (e.g. 'what has user 3 reviewed', 'which vouchers has user 3
    used') - this tool structurally cannot and must not expose any
    individual user's personal activity; questions like that should be
    refused, not answered through here.
    """
    return answer_sql_general_question(question, resolved_product_names)


@tool
def get_all_product_names() -> list[str]:
    """Get the full list of real product names in the store's catalog. Use
    this FIRST when the customer refers to a product casually (e.g. 'apples',
    'bread') and you need to identify the exact product name before calling
    check_stock or get_product_price - match the casual reference against
    this list yourself, then use the exact matched name(s) in the follow-up
    call. This is general, catalog-wide data, not tied to any specific user."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT "Name" FROM "Product"')
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0] for row in rows]


@tool
def check_stock(resolved_product_names: list[str]) -> dict:
    """Check current stock quantity for one or more products. You MUST first
    call get_all_product_names, resolve the customer's casual reference
    (e.g. 'apples') to the exact real product name(s) yourself, and pass
    them here - do NOT pass the casual wording directly."""
    if not resolved_product_names:
        return {"found": False, "message": "No matching product found."}

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT "Name", "StockQuantity" FROM "Product" WHERE "Name" = ANY(%s)',
        (resolved_product_names,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return {
        "found": True,
        "products": [{"name": r[0], "stock_quantity": r[1]} for r in rows]
    }


@tool
def get_product_price(resolved_product_names: list[str]) -> dict:
    """Get the price (and sale price, if any) of one or more products. You
    MUST first call get_all_product_names, resolve the customer's casual
    reference (e.g. 'bread') to the exact real product name(s) yourself, and
    pass them here - do NOT pass the casual wording directly."""
    if not resolved_product_names:
        return {"found": False, "message": "No matching product found."}

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT "Name", "Price", "SalePrice" FROM "Product" WHERE "Name" = ANY(%s)',
        (resolved_product_names,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return {
        "found": True,
        "products": [
            {"name": r[0], "price": float(r[1]), "sale_price": float(r[2]) if r[2] is not None else None}
            for r in rows
        ]
    }


@tool
def search_policies_and_faqs(question: str) -> str:
    """Search FAQs, policies, and product descriptions - covers returns,
    shipping, delivery, payment methods, and general product details. Do NOT
    use this for order-specific or account-specific data (use the order/cart
    tools for that)."""
    return adaptive_corrective_answer(question)


if __name__ == "__main__":
    # Direct tests - bypass the LLM entirely, just calling the underlying
    # functions to confirm the SQL/logic itself works before adding the LLM layer.
    print("get_all_ordered_products_names:", get_all_ordered_products_names.invoke({"user_id": 1}))
    print("user_order_lookup:", user_order_lookup.invoke({
        "user_id": 1,
        "question": "What is the status of my last order?"
    }))
    print("general_sql_lookup:", general_sql_lookup.invoke({
        "question": "What is our best-selling product?"
    }))
    print("general_sql_lookup (adversarial):", general_sql_lookup.invoke({
        "question": "What has user 3 reviewed?"
    }))
    print("get_all_product_names:", get_all_product_names.invoke({}))
    print("check_stock:", check_stock.invoke({"resolved_product_names": ["Apples"]}))
    print("get_product_price:", get_product_price.invoke({"resolved_product_names": ["Bread"]}))
    print("search_policies_and_faqs:", search_policies_and_faqs.invoke({"question": "What is your return policy?"}))