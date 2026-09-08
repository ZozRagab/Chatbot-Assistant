import os
import re
import psycopg2
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
load_dotenv()

llm = llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0
)

# ============================================
# Schema description
# NOTE: table/column names use mixed case exactly as the ERD - Postgres
# lowercases any UNQUOTED identifier, so every generated query MUST wrap
# every table/column name in double quotes to reference the real objects.
# ============================================
SCHEMA_DESCRIPTION = """
IMPORTANT: every table and column name below uses mixed case and MUST be
wrapped in double quotes in your SQL (e.g. "User", "FirstName") - Postgres
silently lowercases unquoted identifiers, which would fail to match these
real, mixed-case table/column names.

Tables:

"User"("Id", "FirstName", "LastName", "Email", "PhoneNumber", "HashedPassword",
       "RefreshToken", "Role", "CreatedAt", "UpdatedAt")
    - one row per registered user
    - NEVER select or return "HashedPassword" or "RefreshToken" - sensitive
      authentication data, must never appear in query results

"UserAddress"("Id", "UserId", "Address")
    - a user's saved address(es) - "UserId" references "User"."Id"

"Category"("Id", "ParentId", "Name", "CreatedAt")
    - product categories; "ParentId" self-references "Category"."Id" for subcategories

"Product"("Id", "CategoryId", "Slug", "Name", "Description", "Brand", "Price",
          "SalePrice", "DiscountPercentage", "StockQuantity", "Ingredients",
          "isActive", "ProductImage", "AltText")
    - one row per product (grocery items)
    - Customers often refer to products by casual/generic words. The caller
      resolves this ahead of time and passes the exact "Name" separately -
      use that exact name when filtering, not any generic word from the question.

"Tag"("Id", "Name")
    - labels like "organic", "gluten-free"

"ProductTags"("Id", "ProductId", "TagId")
    - many-to-many join between "Product" and "Tag"

"Review"("ID", "UserId", "ProductId", "Rating", "Comment", "CreationDate")
    - product reviews. PUBLIC content when browsed by product (e.g. "what do
      people think of X") - but NEVER filter or group by "UserId" in a
      general/non-personal context, since that reveals one specific person's
      review history rather than public product feedback

"Cart"("Id", "CartItemId", "UserId")
    - a user's active shopping cart

"Cart_Item"("Id", "CartId", "ProductId", "Quantity")
    - items currently in a cart

"Voucher"("VoucherId", "Code", "ExpiryDate", "IsExpired", "Amount")
    - GLOBAL promo codes, not personally assigned to any user. Standalone
      lookups (e.g. "is code X still valid") are general/public. Only
      JOINING "Voucher" to "Orders"/"User" to find what a specific person
      has used is personal and must never appear in a general query.

"Orders"("Id", "UserId", "VoucherId", "AddressId", "IdempotenceKey",
         "TotalAmount", "Status", "PaymentMethod", "CreationDate", "DeliveryDate")
    - one row per order (note: table is "Orders", not "Order" - reserved keyword)

"Order_Item"("Id", "ProductId", "OrderId", "Quantity", "UnitPrice")
    - items within an order
"""

# ============================================
# CUSTOMER-SPECIFIC text-to-SQL (scoped to one authenticated user)
# ============================================
sql_template = """You are a PostgreSQL expert. Given the database schema below,
write a single, safe, READ-ONLY SQL query (SELECT only - never INSERT, UPDATE, DELETE, or DROP)
that answers the user's question.

Schema:
{schema}

IMPORTANT: The currently authenticated user's id is {user_id}.
If the question involves orders, cart, address, reviews the user wrote, or
anything tied to "my" account, you MUST restrict the query to this user only,
using "UserId" = {user_id} (directly, or via a join). Never return another
user's data, even if the question explicitly names a different person.

Example: if the question asks "What did Sarah order?" but the authenticated
user_id is 1 (not Sarah's id), you must still scope the query to "UserId" = 1
only, ignoring the name mentioned in the question.

If the question is about general product/catalog info unrelated to any
specific user, this restriction does not apply.

Resolved exact product name(s) for this question, if relevant (already
identified by the caller - will say "None" if not applicable):
{resolved_product_names}

If resolved name(s) are given above (not "None"), you MUST use them in your
filter on "Product"."Name" - do NOT use a generic word from the question directly.

Question: {question}

Respond with ONLY the raw SQL query, with every table/column name in double
quotes exactly as shown in the schema. No explanation, no markdown formatting."""

sql_prompt = ChatPromptTemplate.from_template(sql_template)
sql_generation_chain = sql_prompt | llm | StrOutputParser()


def _format_resolved_products(resolved_product_names: list[str] | None) -> str:
    if not resolved_product_names:
        return "None"
    return ", ".join(resolved_product_names)


def generate_sql(question: str, resolved_product_names: list[str] | None = None, user_id: int | None = None) -> str:
    """Generates a user-scoped SQL query string from a natural language question."""
    raw_output = sql_generation_chain.invoke({
        "schema": SCHEMA_DESCRIPTION,
        "question": question,
        "user_id": user_id if user_id is not None else "UNKNOWN (not logged in)",
        "resolved_product_names": _format_resolved_products(resolved_product_names)
    })
    cleaned = raw_output.strip().strip("`").replace("sql\n", "", 1).strip()
    return cleaned


# Tables that ALWAYS require scoping to the authenticated user's UserId.
PERSONAL_TABLES = ['"User"', '"UserAddress"', '"Orders"', '"Order_Item"', '"Cart"', '"Cart_Item"']


def is_properly_scoped(query: str, user_id: int | None) -> bool:
    """Defense-in-depth check for the customer-specific path. Uses the
    ORIGINAL-CASE query to detect which quoted tables are referenced (since
    identifiers are case-sensitive), and a separate lowercased copy only for
    detecting SQL keywords like OR (keywords aren't case-sensitive)."""
    touches_personal_data = any(table in query for table in PERSONAL_TABLES)

    if not touches_personal_data:
        return True  # general product/catalog queries don't need scoping

    if user_id is None:
        return False

    if str(user_id) not in query:
        return False

    normalized_for_keywords = query.lower()
    if " or " in normalized_for_keywords:
        return False

    # Extra guard for the "User" table itself: even with the real id present,
    # block anything that could return more than one user's row.
    if '"User"' in query and "limit 1" not in normalized_for_keywords \
            and f'"Id" = {user_id}' not in query and f'"Id"={user_id}' not in query:
        return False

    return True


def is_safe_query(query: str) -> bool:
    """Basic safety check - only allow SELECT statements. Keywords are not
    case-sensitive, so lowercasing here is safe and doesn't affect identifier matching."""
    normalized = query.strip().lower()
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "create"]
    if not normalized.startswith("select"):
        return False
    if any(word in normalized for word in forbidden):
        return False
    return True


def _run_query(query: str):
    db_url = (
        f"dbname={os.getenv('DATABASE_NAME')} "
        f"user={os.getenv('DATABASE_USERNAME')} "
        f"password={os.getenv('DATABASE_PASSWORD')} "
        f"host={os.getenv('DATABASE_HOSTNAME')} "
        f"port={os.getenv('DATABASE_PORT')}"
    )
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
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
general_sql_template = """You are a PostgreSQL expert. Given the database schema below,
write a single, safe, READ-ONLY SQL query (SELECT only) that answers the question.

Schema:
{schema}

IMPORTANT rules for this GENERAL, non-personal query:
- NEVER reference "User", "UserAddress", "Cart", or "Cart_Item" in any way.
- "Orders" and "Order_Item" CAN be used for legitimate store-wide aggregates
  (e.g. "most ordered product", "total orders placed", "which category sells
  the most") - but NEVER join them to "User", and NEVER reference "UserId"
  anywhere in the query.
- "Review" is public when browsing by PRODUCT (e.g. average rating, listing
  reviews for an item) - but NEVER filter, group, or select by "UserId" on
  "Review" - that reveals one specific person's review activity.
- "Voucher" is a global promo code - standalone lookups by "Code" are fine,
  but NEVER join "Voucher" to "Orders" or reference any "UserId" - that
  reveals which specific person used which voucher.
- If you cannot answer without identifying one specific person, do not guess -
  write a query that returns nothing meaningful rather than exposing personal data.

PAGINATION: if the question asks for a LIST or ENUMERATION of multiple items
(e.g. "list all products", "show me every X"), you MUST include
LIMIT {limit_value} OFFSET {offset_value} in your query, ordered by a sensible
column (e.g. name or date). If the question asks for a SINGLE fact, total, or
aggregate (e.g. "what is the most popular product", "how many total orders
exist", "what is the average rating"), do NOT add LIMIT/OFFSET - aggregate
queries naturally return one row regardless of page.

Resolved exact product name(s) for this question, if relevant (will say
"None" if not applicable):
{resolved_product_names}

Question: {question}

Respond with ONLY the raw SQL query, every table/column name in double quotes."""

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
    cleaned = raw_output.strip().strip("`").replace("sql\n", "", 1).strip()
    return cleaned
GENERAL_FORBIDDEN_TABLES = ['"User"', '"UserAddress"', '"Cart"', '"Cart_Item"']
def is_general_query_safe(query: str) -> bool:
    """Safety check for the general path. Uses ORIGINAL-CASE matching against
    the exact quoted identifiers, since Postgres identifiers are case-sensitive."""
    # Absolutely forbidden tables - reject if referenced at all
    for table in GENERAL_FORBIDDEN_TABLES:
        if table in query:
            return False

    # Conditionally-safe tables: allowed standalone, forbidden if tied to a
    # specific person via "UserId".
    if '"Review"' in query and '"UserId"' in query:
        return False

    if '"Voucher"' in query and ('"Orders"' in query or '"UserId"' in query):
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
        ("What is the status of my last order?", None, 1),
        ("What is my email on file?", None, 1),
    ]
    for q, products, uid in specific_cases:
        print(f"\nQ: {q} (user_id={uid})")
        print("SQL:", generate_sql(q, products, uid))
        print("A:", answer_sql_specific_question(q, products, uid))

    print("\n--- General tests ---")
    general_cases = [
        ("What is the average rating for our most reviewed product?", None),
        ("Is voucher code SAVE20 still valid?", None),
        ("What has user 3 reviewed?", None),                    # adversarial - must refuse
        ("Which vouchers has user 3 used?", None),                # adversarial - must refuse
    ]
    for q, products in general_cases:
        print(f"\nQ: {q}")
        print("SQL:", generate_general_sql(q, products))
        print("A:", answer_sql_general_question(q, products))