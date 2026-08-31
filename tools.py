import os
import psycopg2
from dotenv import load_dotenv
from langchain_core.tools import tool
from A_C_rag import adaptive_corrective_answer
from text_to_sql import answer_sql_specific_question, answer_sql_general_question
from langchain_core.runnables import RunnableConfig
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
def general_sql_lookup(question: str, resolved_product_names: list[str] | None = None, page: int = 1) -> dict:
    """Answer a GENERAL, store-wide question about products, categories,
    reviews (by product, not by person), or vouchers - NOT tied to any
    specific user. Use for questions like 'what's our best-selling product',
    'what's the average rating for X', 'is voucher code SAVE20 still valid',
    or 'list all products'.

    PAGINATION: results are paginated, 50 items per page - check "has_more"
    in the returned dict. Single-fact/aggregate questions (e.g. "most
    popular product") naturally return has_more=False on page 1. For
    LIST-style questions ("list all products"), call this ONCE per
    question, even if has_more is True - do NOT automatically call again
    for the next page within the same turn. Instead, explicitly tell the
    customer more results exist and that they can ask to see more.

    Do NOT use this for anything about the currently logged-in user's own
    orders/cart/account (use user_order_lookup instead), and NEVER use it to
    answer a question about one specific named person's reviews or voucher
    usage (e.g. 'what has user 3 reviewed', 'which vouchers has user 3
    used') - this tool structurally cannot and must not expose any
    individual user's personal activity; questions like that should be
    refused, not answered through here.
    """
    return answer_sql_general_question(question, resolved_product_names, page)

@tool
def get_all_product_names(page: int = 1) -> dict:
    """Get the store's catalog product names - PAGINATED, 50 names per page.
    Use this FIRST when the customer refers to a product casually (e.g.
    'apples', 'bread') and you need to identify the exact product name
    before calling other tools - match the casual reference against the
    returned names yourself.

    PAGINATION AND SEARCHING ACROSS PAGES:
    - If you find a confident match on this page, stop - no need to fetch
    further pages.
    - If you do NOT find a match on this page AND has_more is True, you
    MUST call again with page+1 to keep searching - a single empty page
    does NOT mean the product doesn't exist, since names are spread
    across multiple pages.
    - If has_more becomes False and you have STILL found no match after
    checking every page, tell the customer honestly that you could not
    find a matching product in the catalog - do NOT guess, substitute
    an unrelated product, or imply a match exists when it doesn't.
    """
    page_size = 50   
    offset = (page - 1) * page_size

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT DISTINCT "Name" FROM "Product" ORDER BY "Name" LIMIT %s OFFSET %s',
        (page_size + 1, offset)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    has_more = len(rows) > page_size
    names = [row[0] for row in rows[:page_size]]

    return {"items": names, "page": page, "has_more": has_more}

@tool
async def sql_agent_tool(question: str, config: RunnableConfig) -> str:
    """Delegate a question about products, orders, cart, stock, prices,
    reviews, or vouchers to the specialized SQL data agent. Use this for
    ANY question requiring structured store/order data - it handles product
    name resolution and pagination internally and returns one final answer.
    Do NOT use this for policy/FAQ/general knowledge questions."""
    from sql_ReAct import c_graph as compiled_graph
    user_id = config["configurable"]["user_id"]

    sub_config = {"configurable": {"user_id": user_id}}

    result = await compiled_graph.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config=sub_config
    )
    return result["messages"][-1].content
@tool
def search_policies_and_faqs(question: str) -> str:
    """Search FAQs, policies, and product descriptions - covers returns,
    shipping, delivery, payment methods, and general product details. Do NOT
    use this for order-specific or account-specific data (use the order/cart
    tools for that)."""
    return adaptive_corrective_answer(question)


