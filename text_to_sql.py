import re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from db import connection
load_dotenv()

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0
)

# ============================================
# Schema description
# Deliberately lists ONLY the tables/columns the fallback queries have a
# reason to touch. Backend-internal columns (Guid, Slug, IdempotencyKey,
# image fields, audit timestamps) and auth/infra tables (RefreshTokens,
# VerificationTokens, ServiceClients, UserActivities) are left out on purpose
# so the model never learns to reach for them; the guards below hard-reject
# them as well.
# ============================================
SCHEMA_DESCRIPTION = """
Dialect: Microsoft SQL Server (T-SQL).
- For a limited number of rows use TOP (n), or ORDER BY ... OFFSET n ROWS
  FETCH NEXT m ROWS ONLY. NEVER use LIMIT.
- The orders table is a reserved word: ALWAYS write it as [Order].
- No other identifier needs quoting. Use the names exactly as listed.

Tables:

Users(Id, FirstName, LastName, Email, PhoneNumber)
    - one row per registered customer

UserAddresses(Id, UserId, Location)
    - a customer's saved delivery address(es); Location is free text

Categories(Id, Name)
    - exactly three real categories: 'Fruits', 'vegetables', 'Packages'
      ('Packages' = everything packaged: dairy, bakery, drinks, pantry, snacks)

Products(Id, CategoryId, Name, Description, Price, StockQuantity)
    - one row per product. Customers often use casual words; the caller
      resolves those to the exact Name ahead of time and passes it separately -
      filter on that exact Name, never on a generic word from the question.

Tags(Id, Name) and ProductTags(ProductId, TagId)
    - product labels (e.g. 'Milk', 'Apple'). The tag 'sales_deals' marks a
      product as currently on sale; no discount percentage is stored anywhere.

ProductReviews(Id, UserId, ProductId, Rating, Comment, CreatedAt)
    - Rating is an int. PUBLIC when browsed by product (e.g. "what do people
      think of X") - but NEVER filter or group by UserId in a general,
      non-personal query; that exposes one specific person's review history.

Cart(Id, UserId) and CartItem(Id, CartId, ProductId, Quantity, UnitPrice)
    - a customer's current basket

[Order](Id, UserId, OrderNumber, Address, TotalAmount, Status, CreationDate, DeliveryTime)
    - one row per order. Status is one of: 'Pending', 'Placed', 'Shipped',
      'Delivered', 'Cancelled'

OrderItem(Id, OrderId, ProductId, ProductName, Quantity, UnitPrice)
    - line items; ProductName is the product's name at time of purchase

There is no voucher / promo-code table - vouchers are not implemented. Never
reference one.
"""

# ============================================
# CUSTOMER-SPECIFIC text-to-SQL (scoped to one authenticated user)
# ============================================
sql_template = """You are a Microsoft SQL Server (T-SQL) expert. Given the database schema below,
write a single, safe, READ-ONLY SQL query (SELECT only - never INSERT, UPDATE, DELETE, or DROP)
that answers the user's question.

Schema:
{schema}

IMPORTANT: The currently authenticated user's id is {user_id}.
If the question involves orders, cart, address, reviews the user wrote, or
anything tied to "my" account, you MUST restrict the query to this user only,
using UserId = {user_id} (directly, or via a join). Never return another
user's data, even if the question explicitly names a different person.

Example: if the question asks "What did Sarah order?" but the authenticated
user_id is 1 (not Sarah's id), you must still scope the query to UserId = 1
only, ignoring the name mentioned in the question.

If the question is about general product/catalog info unrelated to any
specific user, this restriction does not apply.

Resolved exact product name(s) for this question, if relevant (already
identified by the caller - will say "None" if not applicable):
{resolved_product_names}

If resolved name(s) are given above (not "None"), you MUST use them in your
filter on Products.Name - do NOT use a generic word from the question directly.

Question: {question}

Respond with ONLY the raw SQL query - write the orders table as [Order], every
other identifier unquoted exactly as in the schema. No explanation, no markdown."""

sql_prompt = ChatPromptTemplate.from_template(sql_template)
sql_generation_chain = sql_prompt | llm | StrOutputParser()


def _format_resolved_products(resolved_product_names: list[str] | None) -> str:
    if not resolved_product_names:
        return "None"
    return ", ".join(resolved_product_names)


def _clean_sql(raw_output: str) -> str:
    return raw_output.strip().strip("`").replace("sql\n", "", 1).strip()


def generate_sql(question: str, resolved_product_names: list[str] | None = None, user_id: int | None = None) -> str:
    """Generates a user-scoped SQL query string from a natural language question."""
    raw_output = sql_generation_chain.invoke({
        "schema": SCHEMA_DESCRIPTION,
        "question": question,
        "user_id": user_id if user_id is not None else "UNKNOWN (not logged in)",
        "resolved_product_names": _format_resolved_products(resolved_product_names)
    })
    return _clean_sql(raw_output)


# ============================================
# Guards
# T-SQL identifiers aren't case-sensitive and may appear bare, bracketed,
# quoted, or dbo.-prefixed, so table/column detection is a whole-word,
# case-insensitive match on a copy with that punctuation stripped. Whole-word
# matters: [Order] must not match OrderItem, Users must not match UserAddresses.
# ============================================
def _strip_identifier_noise(query: str) -> str:
    stripped = re.sub(r'[\[\]"]', "", query)
    return re.sub(r"\bdbo\.", "", stripped, flags=re.I)


def _mentions(query: str, identifier: str) -> bool:
    return re.search(rf"\b{re.escape(identifier)}\b", _strip_identifier_noise(query), re.I) is not None


# Tables that ALWAYS require scoping to the authenticated user's UserId.
PERSONAL_TABLES = ["Users", "UserAddresses", "Order", "OrderItem", "Cart", "CartItem"]

# Never legitimate in any generated query: auth/infra tables and the one
# sensitive column on Users. Not listed in the schema, hard-rejected here.
SENSITIVE_IDENTIFIERS = [
    "HashedPassword", "RefreshTokens", "VerificationTokens", "ServiceClients",
    "UserActivities", "__EFMigrationsHistory",
]


def is_properly_scoped(query: str, user_id: int | None) -> bool:
    """Defense-in-depth check for the customer-specific path."""
    touches_personal_data = any(_mentions(query, table) for table in PERSONAL_TABLES)

    if not touches_personal_data:
        return True  # general product/catalog queries don't need scoping

    if user_id is None:
        return False

    if str(user_id) not in query:
        return False

    normalized_for_keywords = query.lower()
    if " or " in normalized_for_keywords:
        return False

    # Extra guard for the Users table itself: even with the real id present,
    # block anything that could return more than one user's row.
    if _mentions(query, "Users"):
        pinned_to_self = re.search(rf"\bId\s*=\s*{user_id}\b", _strip_identifier_noise(query), re.I)
        single_row = re.search(r"\btop\s*\(?\s*1\s*\)?", normalized_for_keywords)
        if not pinned_to_self and not single_row:
            return False

    return True


def is_safe_query(query: str) -> bool:
    """Only allow a single SELECT statement, with no write/exec keywords and
    no sensitive identifiers.

    Keyword checks are whole-word: a plain substring test would reject any
    query touching CreatedAt (contains 'create') or an UpdatedAt column."""
    normalized = query.strip().lower()
    if not normalized.startswith("select"):
        return False
    # one statement only - a trailing ';' is fine, anything after one is not
    if ";" in normalized.rstrip().rstrip(";"):
        return False
    forbidden = (r"\b(insert|update|delete|drop|alter|truncate|create|merge|grant|revoke|"
                 r"exec|execute|openrowset|opendatasource|xp_\w+|sp_\w+)\b")
    if re.search(forbidden, normalized):
        return False
    if any(_mentions(query, ident) for ident in SENSITIVE_IDENTIFIERS):
        return False
    return True


def _run_query(query: str):
    try:
        # Pooled checkout (see db.py). The cursor is closed in `finally` so the
        # result set is released before the connection goes back to the pool.
        with connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                rows = [tuple(r) for r in cursor.fetchall()]
            finally:
                cursor.close()
        return {"columns": columns, "rows": rows}, None
    except Exception as e:
        return None, f"Query execution failed: {e}"


def execute_sql(query: str, user_id: int | None):
    if not is_safe_query(query):
        return None, "Query rejected: only SELECT statements are allowed."
    if not is_properly_scoped(query, user_id):
        print(f"  [DEBUG] Query REJECTED by scoping check: {query!r}")
        return None, "Query rejected: this request requires authentication and could not be safely scoped to your account."
    return _run_query(query)


DESTRUCTIVE_INTENT_PATTERNS = [
    r"\bcancel\b", r"\bdelete\b", r"\bremove\b",
    r"\bmodify\b", r"\bupdate\b", r"\bchange\b.*\bmy\b",
]


def has_destructive_intent(question: str) -> bool:
    normalized = question.lower()
    return any(re.search(pattern, normalized) for pattern in DESTRUCTIVE_INTENT_PATTERNS)


def _summarize(question: str, rows_result: dict, user_id: int | None) -> str:
    formatted_rows = "\n".join(str(dict(zip(rows_result["columns"], row))) for row in rows_result["rows"])
    summarize_template = """Given this question and the raw database results below,
answer the question in a natural, friendly sentence.

IMPORTANT: These results are strictly scoped to the currently authenticated user's
own account (user_id={user_id}), regardless of any other name mentioned in the
question. If the question asks about a different named person, these results are
actually the authenticated user's OWN data - make this clear rather than implying
the results belong to the person named in the question.

Question: {question}

Database results (belonging to the authenticated user only):
{results}

Answer:"""
    summarize_prompt = ChatPromptTemplate.from_template(summarize_template)
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    return summarize_chain.invoke({
        "question": question, "results": formatted_rows,
        "user_id": user_id if user_id is not None else "N/A"
    })


def answer_sql_specific_question(question: str, resolved_product_names: list[str] | None = None, user_id: int | None = None) -> str:
    if has_destructive_intent(question):
        return ("I'm a read-only assistant and can't delete, cancel, or modify orders. "
                "Please contact customer support directly for that request.")

    query = generate_sql(question, resolved_product_names, user_id)
    result, error = execute_sql(query, user_id)

    if error:
        return f"Sorry, I couldn't retrieve that information. ({error})"
    if not result["rows"]:
        return "No matching records were found."

    return _summarize(question, result, user_id)


# ============================================
# GENERAL / AGGREGATE text-to-SQL (no single user - store-wide / public data)
# ============================================
general_sql_template = """You are a Microsoft SQL Server (T-SQL) expert. Given the database schema below,
write a single, safe, READ-ONLY SQL query (SELECT only) that answers the question.

Schema:
{schema}

IMPORTANT rules for this GENERAL, non-personal query:
- NEVER reference Users, UserAddresses, Cart, or CartItem in any way.
- [Order] and OrderItem CAN be used for legitimate store-wide aggregates
  (e.g. "most ordered product", "total orders placed", "which category sells
  the most") - but NEVER join them to Users, and NEVER reference UserId
  anywhere in the query.
- ProductReviews is public when browsing by PRODUCT (e.g. average rating,
  listing reviews for an item) - but NEVER filter, group, or select by UserId
  on ProductReviews - that reveals one specific person's review activity.
- There is no voucher table - never reference one.
- If you cannot answer without identifying one specific person, do not guess -
  write a query that returns nothing meaningful rather than exposing personal data.

PAGINATION: if the question asks for a LIST or ENUMERATION of multiple items
(e.g. "list all products", "show me every X"), you MUST order by a sensible
column (e.g. name or date) and end the query with
OFFSET {offset_value} ROWS FETCH NEXT {limit_value} ROWS ONLY.
If the question asks for a SINGLE fact, total, or aggregate (e.g. "what is the
most popular product", "how many total orders exist", "what is the average
rating"), do NOT paginate - use TOP (1) where one row is wanted.

Resolved exact product name(s) for this question, if relevant (will say
"None" if not applicable):
{resolved_product_names}

Question: {question}

Respond with ONLY the raw SQL query - write the orders table as [Order], every
other identifier unquoted exactly as in the schema. No explanation, no markdown."""

general_sql_prompt = ChatPromptTemplate.from_template(general_sql_template)
general_sql_generation_chain = general_sql_prompt | llm | StrOutputParser()


GENERAL_PAGE_SIZE = 50

def generate_general_sql(question: str, resolved_product_names: list[str] | None = None, page: int = 1) -> str:
    offset_value = (page - 1) * GENERAL_PAGE_SIZE
    limit_value = GENERAL_PAGE_SIZE + 1

    raw_output = general_sql_generation_chain.invoke({
        "schema": SCHEMA_DESCRIPTION,
        "question": question,
        "resolved_product_names": _format_resolved_products(resolved_product_names),
        "offset_value": offset_value,
        "limit_value": limit_value
    })
    return _clean_sql(raw_output)


# Voucher: the table no longer exists, so any reference is invalid - a hard
# reject is both safer and a clearer failure than letting it hit the DB.
GENERAL_FORBIDDEN_TABLES = ["Users", "UserAddresses", "Cart", "CartItem", "Voucher", "Vouchers"]

def is_general_query_safe(query: str) -> bool:
    """Safety check for the general path."""
    # Absolutely forbidden tables - reject if referenced at all
    if any(_mentions(query, table) for table in GENERAL_FORBIDDEN_TABLES):
        return False

    # Conditionally-safe table: allowed standalone, forbidden if tied to a
    # specific person via UserId.
    if _mentions(query, "ProductReviews") and _mentions(query, "UserId"):
        return False

    return True


def execute_general_sql(query: str):
    if not is_safe_query(query):
        return None, "Query rejected: only SELECT statements are allowed."
    if not is_general_query_safe(query):
        print(f"  [DEBUG] General query REJECTED by safety check: {query!r}")
        return None, "Query rejected: general queries must not access individual user data."
    return _run_query(query)


def answer_sql_general_question(question: str,  resolved_product_names: list[str] | None = None, page: int| None = 1) -> str:
    if has_destructive_intent(question):
        return ("I'm a read-only assistant and can't delete, cancel, or modify orders. "
                "Please contact customer support directly for that request.")

    query = generate_general_sql(question, resolved_product_names, page)
    result, error = execute_general_sql(query)

    if error:
        return f"Sorry, I couldn't retrieve that information. ({error})"
    if not result["rows"]:
        return "No matching records were found."

    formatted_rows = "\n".join(str(dict(zip(result["columns"], row))) for row in result["rows"])
    summarize_template = """Given this question and the raw database results below,
answer the question in a natural, friendly sentence.

Question: {question}

Database results:
{results}

Answer:"""
    summarize_prompt = ChatPromptTemplate.from_template(summarize_template)
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    return summarize_chain.invoke({"question": question, "results": formatted_rows})


if __name__ == "__main__":
    print("--- Specific (personal) tests ---")
    specific_cases = [
        ("What is the status of my last order?", None, 122),
        ("What is my email on file?", None, 122),
    ]
    for q, products, uid in specific_cases:
        print(f"\nQ: {q} (user_id={uid})")
        print("SQL:", generate_sql(q, products, uid))
        print("A:", answer_sql_specific_question(q, products, uid))

    print("\n--- General tests ---")
    general_cases = [
        ("How many products are in each category?", None),
        ("What has user 3 reviewed?", None),                    # adversarial - must refuse
    ]
    for q, products in general_cases:
        print(f"\nQ: {q}")
        print("SQL:", generate_general_sql(q, products))
        print("A:", answer_sql_general_question(q, products))
