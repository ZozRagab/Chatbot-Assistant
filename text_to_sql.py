import os
import re
import psycopg2
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ============================================
# Schema description - given to the LLM so it knows
# what tables/columns exist and how they relate
# ============================================
SCHEMA_DESCRIPTION = """
Tables:

products(id, name, category, price)
    - one row per product
    - Customers often refer to products by generic category words ("laptop",
      "the phone I ordered") rather than the exact product name. The caller
      (the agent) resolves this ahead of time and passes the exact product
      name(s) separately - use those exact names when filtering, not any
      generic word that might appear in the question itself.

customers(id, name, email, hashed_password)
    - one row per customer
    - NEVER select or return hashed_password in any query - it is sensitive
      authentication data and must never appear in query results
    - NEVER return more than the authenticated customer's own row from this
      table. Never list, enumerate, or return other customers' names or
      emails under any circumstance, even if explicitly asked to "show all
      customers" or similar - always filter to id = the authenticated
      customer_id when this table is involved

orders(id, customer_id, order_date, status)
    - one row per order (a single checkout event)
    - customer_id references customers.id
    - status is one of: 'processing', 'shipped', 'delivered'

order_items(id, order_id, product_id, quantity)
    - one row per product within an order (an order can have multiple items)
    - order_id references orders.id
    - product_id references products.id

inventory(id, product_id, size, color, qty)
    - stock quantity per product variant
    - product_id references products.id
    - size/color can be NULL for products without variants
"""


def _format_resolved_products(resolved_product_names: list[str] | None) -> str:
    """Turns a list of resolved product names (or None) into the plain-text
    form inserted into the prompt."""
    if not resolved_product_names:
        return "None"
    return ", ".join(resolved_product_names)


# ============================================
# CUSTOMER-SPECIFIC text-to-SQL (scoped to one authenticated customer)
# ============================================
sql_template = """You are a PostgreSQL expert. Given the database schema below,
write a single, safe, READ-ONLY SQL query (SELECT only - never INSERT, UPDATE, DELETE, or DROP)
that answers the user's question.

Schema:
{schema}

IMPORTANT: The currently authenticated customer's id is {customer_id}.
If the question involves orders, order history, or anything tied to "my" account,
you MUST restrict the query to this customer only, using orders.customer_id = {customer_id}
(directly, or via a join). Never return another customer's data, even if the
question explicitly names a different customer or asks about "someone else's" orders.

Example: if the question asks "What did Omar Hassan order?" but the authenticated
customer_id is 1 (not Omar's id), you must still scope the query to customer_id = 1
only, ignoring the name mentioned in the question. Do not look up or filter by
a different customer's name under any circumstance.

IMPORTANT: When filtering by product name, use case-insensitive partial matching
with ILIKE and % wildcards (e.g. p.name ILIKE '%AeroBook%'), NOT an exact
equality match (=). Exact string matches will almost always fail.

Resolved exact product name(s) for this question (already identified by the
caller from the customer's real order history - will say "None" if the
question isn't about a specific product). If MORE THAN ONE name is listed,
it means the casual reference was ambiguous - use = ANY(array) matching
against all of them, and let ORDER BY / LIMIT resolve which one is actually
correct (e.g. "the LAST one I ordered" -> ORDER BY o.order_date DESC LIMIT 1):
{resolved_product_names}

If resolved name(s) are given above (not "None"), you MUST use them in your
filter - do NOT use a generic word taken directly from the question (e.g. do
not use '%laptop%' if "AeroBook Pro 14" was resolved above; use '%AeroBook%' instead).

Question: {question}

Respond with ONLY the raw SQL query. No explanation, no markdown formatting, no backticks."""

sql_prompt = ChatPromptTemplate.from_template(sql_template)
sql_generation_chain = sql_prompt | llm | StrOutputParser()


def generate_sql(question: str, resolved_product_names: list[str] | None = None, customer_id: int | None = None) -> str:
    """Generates a customer-scoped SQL query string from a natural language
    question. resolved_product_names should be the exact product name(s)
    already identified by the caller (e.g. via get_all_ordered_products_names
    + the agent's own reasoning) - pass more than one if the casual reference
    was ambiguous between several real products; this function no longer
    resolves casual product references itself."""
    raw_output = sql_generation_chain.invoke({
        "schema": SCHEMA_DESCRIPTION,
        "question": question,
        "customer_id": customer_id if customer_id is not None else "UNKNOWN (not logged in)",
        "resolved_product_names": _format_resolved_products(resolved_product_names)
    })
    cleaned = raw_output.strip().strip("`").replace("sql\n", "", 1).strip()
    return cleaned


def is_properly_scoped(query: str, customer_id: int | None) -> bool:
    """Defense-in-depth check: if this query touches orders/order_items/customers
    and a customer_id was provided, the query MUST reference that specific
    customer_id AND must not use OR logic that could bypass scoping.
    This does not rely on trusting the LLM alone - it verifies the actual SQL text.

    NOTE: 'customers' was added to this list after adversarial testing found
    that a query touching ONLY the customers table (e.g. selecting emails for
    every customer) bypassed scoping entirely, since it contained neither
    'orders' nor 'order_items' - a real, serious gap found via a smuggled
    second request in the test suite."""
    normalized = query.lower()
    touches_customer_data = (
        "orders" in normalized
        or "order_items" in normalized
        or "customers" in normalized
    )

    if not touches_customer_data:
        return True

    if customer_id is None:
        return False

    if str(customer_id) not in normalized:
        return False

    if " or " in normalized:
        return False

    if "customers" in normalized and "limit 1" not in normalized and f"id = {customer_id}" not in normalized and f"id={customer_id}" not in normalized:
        return False

    return True


def is_safe_query(query: str) -> bool:
    """Basic safety check - only allow SELECT statements."""
    normalized = query.strip().lower()
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "create"]
    if not normalized.startswith("select"):
        return False
    if any(word in normalized for word in forbidden):
        return False
    return True


def execute_sql(query: str, customer_id: int | None):
    """Executes a customer-scoped SQL query against Postgres and returns the
    results. Refuses to run anything touching order data that isn't properly
    scoped to the authenticated customer."""
    if not is_safe_query(query):
        return None, "Query rejected: only SELECT statements are allowed."

    if not is_properly_scoped(query, customer_id):
        print(f"  [DEBUG] Query REJECTED by scoping check: {query!r}")
        return None, "Query rejected: this request requires authentication and could not be safely scoped to your account."

    return _run_query(query)


def _run_query(query: str):
    """Shared low-level execution, used by both the customer-scoped and
    general-query paths, once each has passed its own safety checks."""
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


# Word-boundary regex matching on core action VERBS, not rigid exact phrases.
# This catches "cancel my last order", "please cancel it", etc. - not just
# the literal phrase "cancel my order" - while \b prevents false positives
# like "cancellation policy" (which contains "cancel" as a substring but is
# a legitimate informational question, not a destructive request).
DESTRUCTIVE_INTENT_PATTERNS = [
    r"\bcancel\b",
    r"\bdelete\b",
    r"\bremove\b",
    r"\bmodify\b",
    r"\bupdate\b",
    r"\bchange\b.*\bmy\b",
]


def has_destructive_intent(question: str) -> bool:
    """Checks if the user's question is asking for a write/delete action,
    not just an information lookup. This must run BEFORE SQL generation -
    checking only the generated SQL's syntax isn't enough, since the LLM
    can (correctly) generate a safe SELECT while the final natural-language
    answer still ends up phrased as if it performed the requested action."""
    normalized = question.lower()
    return any(re.search(pattern, normalized) for pattern in DESTRUCTIVE_INTENT_PATTERNS)


def _summarize(question: str, rows_result: dict, customer_id: int | None) -> str:
    """Shared summarization step for the customer-scoped path - explicitly
    aware of the authenticated identity so it never misattributes results
    to a different named person mentioned in the question."""
    formatted_rows = "\n".join(str(dict(zip(rows_result["columns"], row))) for row in rows_result["rows"])

    summarize_template = """Given this question and the raw database results below,
answer the question in a natural, friendly sentence.

IMPORTANT: These results are strictly scoped to the currently authenticated customer's
own account (customer_id={customer_id}), regardless of any other name mentioned in the
question. If the question asks about a different named person, these results are
actually the authenticated customer's OWN data, NOT that other person's data - make
this clear in your answer rather than implying the results belong to the person named
in the question. Do not attribute this data to any other named individual.

Question: {question}

Database results (belonging to the authenticated customer only):
{results}

Answer:"""
    summarize_prompt = ChatPromptTemplate.from_template(summarize_template)
    summarize_chain = summarize_prompt | llm | StrOutputParser()

    return summarize_chain.invoke({
        "question": question,
        "results": formatted_rows,
        "customer_id": customer_id if customer_id is not None else "N/A"
    })


def answer_sql_specific_question(question: str, resolved_product_names: list[str] | None = None, customer_id: int | None = None) -> str:
    """Full customer-scoped pipeline: question -> generated SQL -> executed
    -> plain-language answer. customer_id must be the authenticated user's
    real id (never trusted from anywhere else). resolved_product_names should
    be the exact product name(s) already identified by the caller, if the
    question references a specific product."""
    if has_destructive_intent(question):
        return ("I'm a read-only assistant and can't delete, cancel, or modify orders. "
                "Please contact customer support directly for that request.")

    query = generate_sql(question, resolved_product_names, customer_id)
    result, error = execute_sql(query, customer_id)

    if error:
        return f"Sorry, I couldn't retrieve that information. ({error})"

    if not result["rows"]:
        return "No matching records were found."

    return _summarize(question, result, customer_id)


# ============================================
# GENERAL / AGGREGATE text-to-SQL (no single customer - store-wide statistics)
# ============================================
general_sql_template = """You are a PostgreSQL expert. Given the database schema below,
write a single, safe, READ-ONLY SQL query (SELECT only - never INSERT, UPDATE, DELETE, or DROP)
that answers the user's question.

Schema:
{schema}

IMPORTANT: This is a GENERAL, store-wide question - NOT tied to any specific
customer. You MUST NOT reference the customers table in any way, and you MUST
NEVER include customer_id anywhere in the query, not even in a WHERE filter
or a GROUP BY - not even if the question names a specific person. If the
question touches orders or order_items, it must be answered as ONE single
AGGREGATE across ALL customers combined (e.g. COUNT, SUM, AVG) - never a list
of individual, identifiable rows, and never a per-customer breakdown.

Example: if asked "how much has customer 3 spent" or "what did Sarah order
the most", these are NOT valid general questions - they ask about one specific
individual. If you cannot answer a question without filtering to one customer,
write a query that returns nothing meaningful rather than exposing individual
data (this tool is only for questions like "what is our best-selling product"
or "how many orders have we processed in total").

Resolved exact product name(s) for this question, if relevant (already
identified by the caller - will say "None" if not applicable):
{resolved_product_names}

Question: {question}

Respond with ONLY the raw SQL query. No explanation, no markdown formatting, no backticks."""

general_sql_prompt = ChatPromptTemplate.from_template(general_sql_template)
general_sql_generation_chain = general_sql_prompt | llm | StrOutputParser()


def generate_general_sql(question: str, resolved_product_names: list[str] | None = None) -> str:
    """Generates a general, store-wide (non-customer-scoped) SQL query,
    e.g. for questions like 'what is the most ordered product' or
    'how many total orders have been placed'."""
    raw_output = general_sql_generation_chain.invoke({
        "schema": SCHEMA_DESCRIPTION,
        "question": question,
        "resolved_product_names": _format_resolved_products(resolved_product_names)
    })
    cleaned = raw_output.strip().strip("`").replace("sql\n", "", 1).strip()
    return cleaned


def is_general_query_safe(query: str) -> bool:
    """Safety check for the general/aggregate path: the customers table must
    NEVER appear at all, and any query touching orders/order_items must be a
    genuine aggregate (contain at least one aggregate function) AND must not
    filter by any specific customer_id value - never a query returning raw,
    individually-identifiable rows, and never an aggregate scoped down to
    one customer (e.g. SUM(...) WHERE customer_id = 3 is still personal,
    even though it uses an aggregate function - this was a real gap found
    after building the first version of this check)."""
    normalized = query.lower()

    if "customers" in normalized:
        return False

    touches_order_data = "orders" in normalized or "order_items" in normalized
    if touches_order_data:
        aggregate_functions = ["count(", "sum(", "avg(", "max(", "min("]
        if not any(fn in normalized for fn in aggregate_functions):
            return False

        # An aggregate function alone doesn't guarantee the result isn't
        # scoped to or broken out by individual customer - explicitly reject
        # any reference to customer_id at all in this path. Note: GROUP BY
        # customer_id is NOT an exception here - it would return every
        # individual customer's own aggregate side by side in one result,
        # which is arguably worse exposure than a single WHERE filter.
        if "customer_id" in normalized:
            return False

    return True


def execute_general_sql(query: str):
    """Executes a general/aggregate SQL query against Postgres. Refuses
    anything touching the customers table, and refuses non-aggregate queries
    that touch order data."""
    if not is_safe_query(query):
        return None, "Query rejected: only SELECT statements are allowed."

    if not is_general_query_safe(query):
        print(f"  [DEBUG] General query REJECTED by safety check: {query!r}")
        return None, "Query rejected: general queries must not access individual customer data and must aggregate order data."

    return _run_query(query)


def answer_sql_general_question(question: str, resolved_product_names: list[str] | None = None) -> str:
    """Full general/store-wide pipeline: question -> generated SQL -> executed
    -> plain-language answer. Never touches any individual customer's data -
    use answer_sql_specific_question instead for anything about the logged-in
    customer's own account."""
    if has_destructive_intent(question):
        return ("I'm a read-only assistant and can't delete, cancel, or modify orders. "
                "Please contact customer support directly for that request.")

    query = generate_general_sql(question, resolved_product_names)
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
    print("--- Customer-specific tests ---")
    specific_cases = [
        ("What is the status of my last order?", None, 1),
        ("Is my laptop under warranty?", ["AeroBook Pro 14"], 2),
        ("What is the status of my last order?", None, None),  # not logged in - should refuse
    ]
    for q, products, cid in specific_cases:
        print(f"\nQ: {q} (customer_id={cid}, resolved_product_names={products})")
        print("SQL:", generate_sql(q, products, cid))
        print("A:", answer_sql_specific_question(q, products, cid))

    print("\n--- General/aggregate tests ---")
    general_cases = [
        ("What is the most ordered product?", None),
        ("How many total orders have been placed across the store?", None),
        ("How much has customer 3 spent in total?", None),  # adversarial - must be refused, not answered
    ]
    for q, products in general_cases:
        print(f"\nQ: {q}")
        print("SQL:", generate_general_sql(q, products))
        print("A:", answer_sql_general_question(q, products))