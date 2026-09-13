from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from db import connection, fetch_all, fetch_on, placeholders
from A_C_rag import adaptive_corrective_answer
from text_to_sql import answer_sql_specific_question, answer_sql_general_question

# All queries below are T-SQL against the backend's SQL Server (see db.py).
# Table names follow the backend's EF Core schema: Products, Categories,
# [Order] (reserved word - always bracketed), OrderItem, Cart, CartItem,
# ProductReviews, UserAddresses, Tags, ProductTags.

PAGE_SIZE = 50

# The store's three real categories. The Categories table also contains
# leftover test rows ('davidfr3f', 'test category') holding junk products
# ('string7', 'davidHANNA3', ...). They cannot simply be deleted: those
# products are referenced by 157 orders and 129 users' carts, and
# Products.CategoryId -> Categories is ON DELETE NO_ACTION, so the delete is
# refused while they exist. Until the backend team cleans them up, the catalog
# is filtered to these three so the assistant never offers a customer a test
# product. Order history is unaffected - OrderItem stores its own ProductName,
# so a past order still shows exactly what was bought.
REAL_CATEGORIES = ("Fruits", "vegetables", "Packages")
_REAL_CATEGORY_PLACEHOLDERS = ", ".join("?" for _ in REAL_CATEGORIES)


@tool
def get_all_ordered_products_names(user_id: int, category: str | None = None) -> list[str]:
    """Distinct product names this user has actually ordered. Use FIRST to
    resolve a casual product reference (e.g. 'my apples') to an exact name
    before calling a follow-up tool - match it yourself against this list,
    then pass ONLY the matching name(s) onward.

    Optional `category` narrows the list to ONE broad category. The store has
    exactly three:
      - 'Fruits'     - fresh fruit
      - 'vegetables' - fresh vegetables
      - 'Packages'   - everything packaged: dairy, bakery, beverages,
                       pantry goods, snacks, frozen
    Pass the string EXACTLY as spelled above (note 'vegetables' is lowercase).
    Use it to cut the list down when the question is clearly about one of the
    three (e.g. 'the milk I bought' -> category='Packages'). Omit it when the
    question spans the whole order history or no category obviously applies.

    Empty list if the user has no orders (or none in that category)."""
    sql = (
        "SELECT DISTINCT p.Name FROM Products p "
        "JOIN OrderItem oi ON oi.ProductId = p.Id "
        "JOIN [Order] o ON o.Id = oi.OrderId "
    )
    if category:
        sql += "JOIN Categories c ON c.Id = p.CategoryId WHERE o.UserId = ? AND c.Name = ?"
        rows = fetch_all(sql, (user_id, category))
    else:
        sql += "WHERE o.UserId = ?"
        rows = fetch_all(sql, (user_id,))
    return [r[0] for r in rows]


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
    reviews by product) not tied to any user - use only if no
    dedicated general tool fits (see system prompt for the list). Writes
    and runs SQL on the fly; mainly for open-ended list/aggregate questions
    that don't match a dedicated tool, e.g. 'list all products'.

    NOT for "what products are in category X" - resolve those with
    get_all_product_names(category=...) and a dedicated tool instead.

    Paginated, 50/page - check has_more, call ONCE per question, and tell
    the customer if more results exist rather than auto-fetching more.

    Never use for the logged-in user's own data (use a personal tool
    instead), and never to expose one specific named person's reviews -
    refuse those instead of attempting them.
    """
    return answer_sql_general_question(question, resolved_product_names, page)


@tool
def get_all_product_names(page: int = 1, category: str | None = None) -> dict:
    """Store's catalog product names - PAGINATED, 50/page. Use FIRST to
    resolve a casual product reference (e.g. 'apples') to an exact name -
    match it yourself against the returned names, then pass ONLY the
    matching name(s) onward to the tool that answers the question.

    Optional `category` narrows the catalog to ONE broad category. The store
    has exactly three:
      - 'Fruits'     - fresh fruit
      - 'vegetables' - fresh vegetables
      - 'Packages'   - everything packaged: dairy, bakery, beverages,
                       pantry goods, snacks, frozen
    Pass the string EXACTLY as spelled above (note 'vegetables' is lowercase).
    Prefer filtering whenever the question points at one of the three - e.g.
    'what dairy do you have' -> category='Packages', then pick out the milk
    /cheese/yoghurt names yourself. It returns far fewer names, so you
    usually avoid paging entirely. Omit it for catalog-wide questions
    ('list all products', 'do you sell umbrellas?').

    Returns names ONLY - no prices. For price/stock/description, pass the
    resolved names to get_product_details. See the system prompt's
    PAGINATION section for the search-across-pages policy."""
    offset = (page - 1) * PAGE_SIZE
    if category:
        rows = fetch_all(
            "SELECT p.Name FROM Products p "
            "JOIN Categories c ON c.Id = p.CategoryId "
            "WHERE c.Name = ? "
            "ORDER BY p.Name OFFSET ? ROWS FETCH NEXT ? ROWS ONLY",
            (category, offset, PAGE_SIZE + 1),
        )
    else:
        # Leftover test categories are excluded - see REAL_CATEGORIES above.
        rows = fetch_all(
            "SELECT p.Name FROM Products p "
            "JOIN Categories c ON c.Id = p.CategoryId "
            f"WHERE c.Name IN ({_REAL_CATEGORY_PLACEHOLDERS}) "
            "ORDER BY p.Name OFFSET ? ROWS FETCH NEXT ? ROWS ONLY",
            (*REAL_CATEGORIES, offset, PAGE_SIZE + 1),
        )
    has_more = len(rows) > PAGE_SIZE
    return {"items": [r[0] for r in rows[:PAGE_SIZE]], "page": page, "has_more": has_more}


# NOTE: get_all_category_names was removed - the store has exactly three fixed
# categories ('Fruits', 'vegetables', 'Packages'), named directly in the
# docstrings above and in the system prompt, so listing them from the DB was a
# wasted tool call.


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
    1 is the one before that, etc. Returns order number, status, total,
    delivery address, dates, and line items. Use for 'last order status',
    'what did I order before that' - map phrasing to the right offset.
    For multiple orders at once, use list_my_orders instead."""
    # Both queries share one connection - opening a second one to the remote
    # server would cost ~0.3s more than the query itself.
    with connection() as conn:
        rows = fetch_on(
            conn,
            "SELECT Id, OrderNumber, Status, TotalAmount, Address, CreationDate, DeliveryTime "
            "FROM [Order] WHERE UserId = ? "
            "ORDER BY CreationDate DESC OFFSET ? ROWS FETCH NEXT 1 ROWS ONLY",
            (user_id, offset),
        )
        if not rows:
            return {"found": False, "message": "No order found at that position - the user may not have that many orders."}

        order_id, order_number, status, total, address, created, delivery = rows[0]
        # OrderItem stores the product name at time of purchase - no join needed.
        item_rows = fetch_on(
            conn,
            "SELECT ProductName, Quantity, UnitPrice FROM OrderItem WHERE OrderId = ?",
            (order_id,),
        )
    return {
        "found": True,
        "order_id": order_id,
        "order_number": order_number,
        "status": status,
        "total_amount": total,
        "delivery_address": address,
        "creation_date": str(created) if created else None,
        "delivery_time": str(delivery) if delivery else None,
        "items": [{"product": r[0], "quantity": r[1], "unit_price": r[2]} for r in item_rows],
    }


@tool
def list_my_orders(user_id: int, page: int = 1) -> dict:
    """Paginated summary list of the user's own past orders (order number,
    status, total, date) - no line items. Use for 'show me my order
    history'. For one order's full detail, use get_order_by_recency instead."""
    offset = (page - 1) * PAGE_SIZE
    rows = fetch_all(
        "SELECT Id, OrderNumber, Status, TotalAmount, CreationDate FROM [Order] "
        "WHERE UserId = ? ORDER BY CreationDate DESC OFFSET ? ROWS FETCH NEXT ? ROWS ONLY",
        (user_id, offset, PAGE_SIZE + 1),
    )
    has_more = len(rows) > PAGE_SIZE
    orders = [
        {"order_id": r[0], "order_number": r[1], "status": r[2],
         "total_amount": r[3], "creation_date": str(r[4])}
        for r in rows[:PAGE_SIZE]
    ]
    return {"items": orders, "page": page, "has_more": has_more}


@tool
def get_cart_contents(user_id: int) -> list[dict]:
    """Everything in the user's cart - product, quantity, unit price. Use for
    'what's in my cart'. Empty list if the cart is empty."""
    rows = fetch_all(
        "SELECT p.Name, ci.Quantity, ci.UnitPrice FROM Cart c "
        "JOIN CartItem ci ON ci.CartId = c.Id "
        "JOIN Products p ON p.Id = ci.ProductId "
        "WHERE c.UserId = ?",
        (user_id,),
    )
    return [{"product": r[0], "quantity": r[1], "unit_price": r[2]} for r in rows]


@tool
def get_saved_addresses(user_id: int) -> list[str]:
    """The user's own saved delivery addresses. Use for 'what's my saved
    address'. Empty list if none are saved - in that case the address used
    for a particular order is available via get_order_by_recency."""
    rows = fetch_all("SELECT Location FROM UserAddresses WHERE UserId = ?", (user_id,))
    return [r[0] for r in rows]


@tool
def get_my_reviews(user_id: int, resolved_product_names: list[str] | None = None) -> list[dict]:
    """Reviews the user themselves wrote - product, rating, comment, date.
    Resolve a casual product name via get_all_ordered_products_names first
    and pass it as resolved_product_names to filter; leave None for all.
    Only ever this user's own reviews - cannot look up another person's."""
    sql = (
        "SELECT p.Name, r.Rating, r.Comment, r.CreatedAt FROM ProductReviews r "
        "JOIN Products p ON p.Id = r.ProductId WHERE r.UserId = ? "
    )
    params: tuple = (user_id,)
    if resolved_product_names:
        sql += f"AND p.Name IN ({placeholders(resolved_product_names)}) "
        params += tuple(resolved_product_names)
    rows = fetch_all(sql + "ORDER BY r.CreatedAt DESC", params)
    return [{"product": r[0], "rating": r[1], "comment": r[2], "date": str(r[3])} for r in rows]


@tool
def get_product_details(resolved_product_names: list[str]) -> list[dict]:
    """Catalog details for named product(s): description, price, stock.
    Covers price/stock/description/'do you sell X' questions in one tool.

    Resolve names via get_all_product_names first (pass its `category` arg
    when the question points at Fruits / vegetables / Packages). This is also
    how you answer "what <category> do you have and what do they cost":
    resolve the names in that category, then pass them all here.

    Empty list = not found - don't guess."""
    if not resolved_product_names:
        return []
    rows = fetch_all(
        f"SELECT Name, Description, Price, StockQuantity FROM Products "
        f"WHERE Name IN ({placeholders(resolved_product_names)})",
        tuple(resolved_product_names),
    )
    return [
        {"name": r[0], "description": r[1], "price": r[2],
         "stock_quantity": r[3], "in_stock": r[3] > 0}
        for r in rows
    ]


# NOTE: get_products_by_category was removed. Category questions are now
# answered by narrowing the name list instead: call get_all_product_names with
# category='Fruits' | 'vegetables' | 'Packages', pick the relevant names, then
# pass those to get_product_details (prices/stock) or whichever tool answers
# the actual question.


@tool
def get_products_on_sale(page: int = 1) -> dict:
    """Paginated list of products currently flagged as deals/on sale, with
    their price. Use for 'what's on sale', 'any deals right now'. The store
    does not record a discount percentage or a previous price - only which
    products are deals."""
    offset = (page - 1) * PAGE_SIZE
    # "On sale" is modelled as the 'sales_deals' tag on a product.
    rows = fetch_all(
        "SELECT p.Name, p.Price FROM Products p "
        "JOIN ProductTags pt ON pt.ProductId = p.Id "
        "JOIN Tags t ON t.Id = pt.TagId "
        "WHERE t.Name = 'sales_deals' "
        "ORDER BY p.Name OFFSET ? ROWS FETCH NEXT ? ROWS ONLY",
        (offset, PAGE_SIZE + 1),
    )
    has_more = len(rows) > PAGE_SIZE
    return {"items": [{"name": r[0], "price": r[1]} for r in rows[:PAGE_SIZE]],
            "page": page, "has_more": has_more}


@tool
def get_best_selling_products(limit: int = 5) -> list[dict]:
    """Top-selling products store-wide by quantity sold. `limit` controls
    how many (default 5; 1 for "THE best seller"). Store-wide aggregate -
    never tied to any user."""
    rows = fetch_all(
        "SELECT TOP (?) p.Name, SUM(oi.Quantity) AS TotalSold "
        "FROM OrderItem oi JOIN Products p ON p.Id = oi.ProductId "
        "GROUP BY p.Name ORDER BY TotalSold DESC",
        (limit,),
    )
    return [{"name": r[0], "total_sold": r[1]} for r in rows]


@tool
def get_top_rated_products(limit: int = 5) -> list[dict]:
    """Highest-rated products store-wide by average rating. `limit`
    controls how many (default 5). Aggregate only - never exposes who
    wrote a review or anything about a specific person."""
    # CAST: Rating is an int and T-SQL's AVG of ints truncates to an int.
    rows = fetch_all(
        "SELECT TOP (?) p.Name, AVG(CAST(r.Rating AS FLOAT)) AS AvgRating, COUNT(*) AS ReviewCount "
        "FROM ProductReviews r JOIN Products p ON p.Id = r.ProductId "
        "GROUP BY p.Name ORDER BY AvgRating DESC, ReviewCount DESC",
        (limit,),
    )
    return [{"name": r[0], "average_rating": float(r[1]), "review_count": r[2]} for r in rows]


@tool
def get_product_reviews(resolved_product_names: list[str], page: int = 1) -> dict:
    """Paginated public reviews (rating, comment, date) for named
    product(s), most recent first. Use for 'what do people think of X',
    'average rating for X' (compute it yourself from the ratings). Resolve
    names via get_all_product_names first. Never includes who wrote a
    review. For the user's OWN reviews, use get_my_reviews instead."""
    if not resolved_product_names:
        return {"items": [], "page": page, "has_more": False}
    offset = (page - 1) * PAGE_SIZE
    rows = fetch_all(
        f"SELECT p.Name, r.Rating, r.Comment, r.CreatedAt FROM ProductReviews r "
        f"JOIN Products p ON p.Id = r.ProductId "
        f"WHERE p.Name IN ({placeholders(resolved_product_names)}) "
        f"ORDER BY r.CreatedAt DESC OFFSET ? ROWS FETCH NEXT ? ROWS ONLY",
        tuple(resolved_product_names) + (offset, PAGE_SIZE + 1),
    )
    has_more = len(rows) > PAGE_SIZE
    items = [{"product": r[0], "rating": r[1], "comment": r[2], "date": str(r[3])}
             for r in rows[:PAGE_SIZE]]
    return {"items": items, "page": page, "has_more": has_more}


# NOTE: check_voucher_validity was removed - the backend dropped the Voucher
# table (vouchers are not being implemented), so there is nothing to query.
# The store's voucher POLICY text still lives in the FAQ/vector store and is
# answered by search_policies_and_faqs, not from the database.


@tool
async def sql_agent_tool(question: str, config: RunnableConfig) -> str:
    """Delegate a question about products, orders, cart, stock, prices, or
    reviews to the specialized SQL data agent. Use this for
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
    # .text (not .content) - this tool is typed to return str, but a Gemini
    # sub-agent llm returns content as a list of blocks, not a plain string.
    return result["messages"][-1].text


@tool
def search_policies_and_faqs(question: str) -> str:
    """Search store policies and FAQs - covers returns, shipping, delivery
    windows, and payment methods. It has NO product data at all: anything
    about a specific product, including its description, belongs to the SQL
    agent. Do NOT use this for order-specific or account-specific data."""
    return adaptive_corrective_answer(question)
