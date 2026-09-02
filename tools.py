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
    """Distinct product names this user has actually ordered. Use FIRST to
    resolve a casual product reference (e.g. 'my apples') to an exact name
    before calling a follow-up tool - match it yourself against this list.
    Empty list if the user has no orders."""
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
    """LAST RESORT for the authenticated user's own orders, order history,
    cart, addresses, or reviews they wrote - use only if no dedicated
    personal tool fits (see system prompt for the list). Writes and runs
    SQL on the fly, so prefer a dedicated tool whenever one applies.

    If a product is referenced casually, resolve it via
    get_all_ordered_products_names first and pass the exact name(s) as
    resolved_product_names - never the casual wording.

    Read-only: refuse any cancel/delete/modify request instead. Never use
    for general/store-wide questions - use general_sql_lookup for those.
    """
    return answer_sql_specific_question(question, resolved_product_names, user_id)

@tool
def general_sql_lookup(question: str, resolved_product_names: list[str] | None = None, page: int = 1) -> dict:
    """LAST RESORT for a general, store-wide question (products, categories,
    reviews by product, vouchers) not tied to any user - use only if no
    dedicated general tool fits (see system prompt for the list). Writes
    and runs SQL on the fly; mainly for open-ended list/aggregate questions
    that don't match a dedicated tool, e.g. 'list all products'.

    Paginated, 50/page - check has_more, call ONCE per question, and tell
    the customer if more results exist rather than auto-fetching more.

    Never use for the logged-in user's own data (use a personal tool
    instead), and never to expose one specific named person's reviews or
    voucher usage - refuse those instead of attempting them.
    """
    return answer_sql_general_question(question, resolved_product_names, page)

@tool
def get_all_product_names(page: int = 1) -> dict:
    """Store's catalog product names - PAGINATED, 50/page. Use FIRST to
    resolve a casual product reference (e.g. 'apples') to an exact name -
    match it yourself against the returned names. See system prompt's
    PAGINATION section for the search-across-pages policy (keep paging
    with page+1 until a match or has_more is False - don't guess)."""
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
def get_all_category_names(page: int = 1) -> dict:
    """Store's category names - PAGINATED, 50/page. Use FIRST to resolve a
    casual category reference (e.g. 'dairy') to an exact name before
    calling get_products_by_category. See system prompt's PAGINATION
    section for the search-across-pages policy."""
    page_size = 50
    offset = (page - 1) * page_size

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT DISTINCT "Name" FROM "Category" ORDER BY "Name" LIMIT %s OFFSET %s',
        (page_size + 1, offset)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    has_more = len(rows) > page_size
    names = [row[0] for row in rows[:page_size]]

    return {"items": names, "page": page, "has_more": has_more}


# ============================================
# DEDICATED FREQUENT-QUESTION TOOLS
#
# Each of these runs a fixed, pre-vetted query instead of asking an LLM to
# write SQL - faster and structurally safe for its one specific purpose.
# They cover the most commonly asked question shapes; anything that doesn't
# fit one of these falls through to user_order_lookup / general_sql_lookup.
#
# Kept GENERIC on purpose: e.g. one order-recency tool with an `offset`
# argument rather than separate "last order" / "order before that" tools.
# ============================================

@tool
def get_order_by_recency(user_id: int, offset: int = 0) -> dict:
    """ONE of the user's own orders by recency: offset=0 is most recent,
    1 is the one before that, etc. Returns status, total, payment method,
    dates, voucher code, and line items. Use for 'last order status',
    'what did I order before that' - map phrasing to the right offset.
    For multiple orders at once, use list_my_orders instead."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT "Orders"."Id", "Orders"."Status", "Orders"."TotalAmount", '
        '"Orders"."PaymentMethod", "Orders"."CreationDate", "Orders"."DeliveryDate", '
        '"Voucher"."Code" '
        'FROM "Orders" '
        'LEFT JOIN "Voucher" ON "Voucher"."VoucherId" = "Orders"."VoucherId" '
        'WHERE "Orders"."UserId" = %s '
        'ORDER BY "Orders"."CreationDate" DESC '
        'LIMIT 1 OFFSET %s',
        (user_id, offset)
    )
    row = cursor.fetchone()
    if row is None:
        cursor.close()
        conn.close()
        return {"found": False, "message": "No order found at that position - the user may not have that many orders."}

    order_id, status, total, payment_method, created, delivered, voucher_code = row
    cursor.execute(
        'SELECT "Product"."Name", "Order_Item"."Quantity", "Order_Item"."UnitPrice" '
        'FROM "Order_Item" JOIN "Product" ON "Product"."Id" = "Order_Item"."ProductId" '
        'WHERE "Order_Item"."OrderId" = %s',
        (order_id,)
    )
    items = [{"product": r[0], "quantity": r[1], "unit_price": r[2]} for r in cursor.fetchall()]
    cursor.close()
    conn.close()

    return {
        "found": True,
        "order_id": order_id,
        "status": status,
        "total_amount": total,
        "payment_method": payment_method,
        "creation_date": str(created) if created else None,
        "delivery_date": str(delivered) if delivered else None,
        "voucher_code": voucher_code,
        "items": items,
    }


@tool
def list_my_orders(user_id: int, page: int = 1) -> dict:
    """Paginated summary list of the user's own past orders (id, status,
    total, date) - no line items. Use for 'show me my order history'. For
    one order's full detail, use get_order_by_recency instead."""
    page_size = 50
    offset = (page - 1) * page_size

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT "Id", "Status", "TotalAmount", "CreationDate" FROM "Orders" '
        'WHERE "UserId" = %s ORDER BY "CreationDate" DESC LIMIT %s OFFSET %s',
        (user_id, page_size + 1, offset)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    has_more = len(rows) > page_size
    orders = [
        {"order_id": r[0], "status": r[1], "total_amount": r[2], "creation_date": str(r[3])}
        for r in rows[:page_size]
    ]
    return {"items": orders, "page": page, "has_more": has_more}


@tool
def get_cart_contents(user_id: int) -> list[dict]:
    """Everything in the user's cart - product, quantity, price. Use for
    'what's in my cart'. Empty list if the cart is empty."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT "Product"."Name", "Cart_Item"."Quantity", "Product"."Price", "Product"."SalePrice" '
        'FROM "Cart" '
        'JOIN "Cart_Item" ON "Cart_Item"."CartId" = "Cart"."Id" '
        'JOIN "Product" ON "Product"."Id" = "Cart_Item"."ProductId" '
        'WHERE "Cart"."UserId" = %s',
        (user_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"product": r[0], "quantity": r[1], "price": r[2], "sale_price": r[3]} for r in rows]


@tool
def get_saved_addresses(user_id: int) -> list[str]:
    """The user's own saved delivery addresses. Use for 'what's my delivery
    address'. Empty list if none are saved."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT "Address" FROM "UserAddress" WHERE "UserId" = %s', (user_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0] for row in rows]


@tool
def get_my_reviews(user_id: int, resolved_product_names: list[str] | None = None) -> list[dict]:
    """Reviews the user themselves wrote - product, rating, comment, date.
    Resolve a casual product name via get_all_ordered_products_names first
    and pass it as resolved_product_names to filter; leave None for all.
    Only ever this user's own reviews - cannot look up another person's."""
    conn = _get_connection()
    cursor = conn.cursor()
    if resolved_product_names:
        cursor.execute(
            'SELECT "Product"."Name", "Review"."Rating", "Review"."Comment", "Review"."CreationDate" '
            'FROM "Review" JOIN "Product" ON "Product"."Id" = "Review"."ProductId" '
            'WHERE "Review"."UserId" = %s AND "Product"."Name" = ANY(%s) '
            'ORDER BY "Review"."CreationDate" DESC',
            (user_id, resolved_product_names)
        )
    else:
        cursor.execute(
            'SELECT "Product"."Name", "Review"."Rating", "Review"."Comment", "Review"."CreationDate" '
            'FROM "Review" JOIN "Product" ON "Product"."Id" = "Review"."ProductId" '
            'WHERE "Review"."UserId" = %s '
            'ORDER BY "Review"."CreationDate" DESC',
            (user_id,)
        )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"product": r[0], "rating": r[1], "comment": r[2], "date": str(r[3])} for r in rows]


@tool
def get_product_details(resolved_product_names: list[str]) -> list[dict]:
    """Catalog details for named product(s): price, sale price, discount,
    stock, brand, description, ingredients, active status. Covers
    price/stock/discount/ingredient/'do you sell X' questions in one tool.
    Resolve names via get_all_product_names first. Empty list = not
    found - don't guess."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT "Name", "Brand", "Price", "SalePrice", "DiscountPercentage", '
        '"StockQuantity", "Description", "Ingredients", "isActive" '
        'FROM "Product" WHERE "Name" = ANY(%s)',
        (resolved_product_names,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [
        {
            "name": r[0], "brand": r[1], "price": r[2], "sale_price": r[3],
            "discount_percentage": r[4], "stock_quantity": r[5],
            "description": r[6], "ingredients": r[7], "is_active": r[8],
        }
        for r in rows
    ]


@tool
def get_products_by_category(resolved_category_names: list[str], page: int = 1) -> dict:
    """Paginated list of active products in named categor(y/ies). Use for
    'what dairy products do you have'. Resolve via get_all_category_names
    first, then pass the exact name(s) as resolved_category_names."""
    page_size = 50
    offset = (page - 1) * page_size

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT "Product"."Name", "Product"."Price", "Product"."SalePrice" '
        'FROM "Product" JOIN "Category" ON "Category"."Id" = "Product"."CategoryId" '
        'WHERE "Category"."Name" = ANY(%s) AND "Product"."isActive" = true '
        'ORDER BY "Product"."Name" LIMIT %s OFFSET %s',
        (resolved_category_names, page_size + 1, offset)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    has_more = len(rows) > page_size
    items = [{"name": r[0], "price": r[1], "sale_price": r[2]} for r in rows[:page_size]]
    return {"items": items, "page": page, "has_more": has_more}


@tool
def get_products_on_sale(page: int = 1) -> dict:
    """Paginated list of active discounted products, highest discount
    first. Use for 'what's on sale', 'any deals right now'."""
    page_size = 50
    offset = (page - 1) * page_size

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT "Name", "Price", "SalePrice", "DiscountPercentage" FROM "Product" '
        'WHERE "DiscountPercentage" > 0 AND "isActive" = true '
        'ORDER BY "DiscountPercentage" DESC LIMIT %s OFFSET %s',
        (page_size + 1, offset)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    has_more = len(rows) > page_size
    items = [
        {"name": r[0], "price": r[1], "sale_price": r[2], "discount_percentage": r[3]}
        for r in rows[:page_size]
    ]
    return {"items": items, "page": page, "has_more": has_more}


@tool
def get_best_selling_products(limit: int = 5) -> list[dict]:
    """Top-selling active products store-wide by quantity sold. `limit`
    controls how many (default 5; 1 for "THE best seller"). Store-wide
    aggregate - never tied to any user."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT "Product"."Name", SUM("Order_Item"."Quantity") AS "TotalSold" '
        'FROM "Order_Item" JOIN "Product" ON "Product"."Id" = "Order_Item"."ProductId" '
        'WHERE "Product"."isActive" = true '
        'GROUP BY "Product"."Name" ORDER BY "TotalSold" DESC LIMIT %s',
        (limit,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"name": r[0], "total_sold": r[1]} for r in rows]


@tool
def get_top_rated_products(limit: int = 5) -> list[dict]:
    """Highest-rated active products store-wide by average rating. `limit`
    controls how many (default 5). Aggregate only - never exposes who
    wrote a review or anything about a specific person."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT "Product"."Name", AVG("Review"."Rating") AS "AvgRating", COUNT(*) AS "ReviewCount" '
        'FROM "Review" JOIN "Product" ON "Product"."Id" = "Review"."ProductId" '
        'WHERE "Product"."isActive" = true '
        'GROUP BY "Product"."Name" ORDER BY "AvgRating" DESC, "ReviewCount" DESC LIMIT %s',
        (limit,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"name": r[0], "average_rating": float(r[1]), "review_count": r[2]} for r in rows]


@tool
def get_product_reviews(resolved_product_names: list[str], page: int = 1) -> dict:
    """Paginated public reviews (rating, comment, date) for named
    product(s), most recent first. Use for 'what do people think of X',
    'average rating for X' (compute it yourself from the ratings). Resolve
    names via get_all_product_names first. Never includes who wrote a
    review. For the user's OWN reviews, use get_my_reviews instead."""
    page_size = 50
    offset = (page - 1) * page_size

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT "Product"."Name", "Review"."Rating", "Review"."Comment", "Review"."CreationDate" '
        'FROM "Review" JOIN "Product" ON "Product"."Id" = "Review"."ProductId" '
        'WHERE "Product"."Name" = ANY(%s) '
        'ORDER BY "Review"."CreationDate" DESC LIMIT %s OFFSET %s',
        (resolved_product_names, page_size + 1, offset)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    has_more = len(rows) > page_size
    items = [
        {"product": r[0], "rating": r[1], "comment": r[2], "date": str(r[3])}
        for r in rows[:page_size]
    ]
    return {"items": items, "page": page, "has_more": has_more}


@tool
def check_voucher_validity(code: str) -> dict:
    """Whether a voucher code exists/is valid - expiry, expired flag,
    amount. Standalone lookup, never joined to orders/users - cannot
    reveal who used a code. Not-found if the code doesn't exist."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT "Code", "ExpiryDate", "IsExpired", "Amount" FROM "Voucher" WHERE "Code" ILIKE %s',
        (code,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if row is None:
        return {"found": False}
    return {"found": True, "code": row[0], "expiry_date": str(row[1]), "is_expired": row[2], "amount": row[3]}


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


