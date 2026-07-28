import os
import psycopg2
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from product_index import find_likely_products

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
      "the phone I ordered") rather than the exact product name. A list of
      likely actual product names for this specific question is provided
      separately below (under "Likely referenced products") - you MUST use
      those exact names when filtering by product, NOT the generic word
      from the question itself.
    - Customers often refer to products by generic category words ("laptop",
      "the phone I ordered") rather than the exact product name. A list of
      likely actual product names for this specific question is provided
      separately below (under "Likely referenced products") - use those
      exact names when filtering, not the generic word from the question.

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

# ============================================
# Text-to-SQL prompt
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

If the question is about general product/inventory info unrelated to any specific
customer, this restriction does not apply.

IMPORTANT: When filtering by product name, use case-insensitive partial matching
with ILIKE and % wildcards (e.g. p.name ILIKE '%AeroBook%'), NOT an exact
equality match (=). Exact string matches will almost always fail.

Likely referenced products for this specific question (found via semantic
search, may be empty if the question isn't about a specific product):
{likely_products}

If this list is non-empty, you MUST use one of these exact product names in
your ILIKE filter - do NOT use a generic word taken directly from the question
(e.g. do not use '%laptop%' if "AeroBook Pro 14" is listed above; use
'%AeroBook%' instead).

Question: {question}

Respond with ONLY the raw SQL query. No explanation, no markdown formatting, no backticks."""

sql_prompt = ChatPromptTemplate.from_template(sql_template)

sql_generation_chain = sql_prompt | llm | StrOutputParser()


def generate_sql(question: str, customer_id: int | None) -> str:
    """Generates a SQL query string from a natural language question,
    scoped to the given customer_id if provided."""
    likely_products = find_likely_products(question)
    likely_products_text = ", ".join(likely_products) if likely_products else "(none found)"

    raw_output = sql_generation_chain.invoke({
        "schema": SCHEMA_DESCRIPTION,
        "question": question,
        "customer_id": customer_id if customer_id is not None else "UNKNOWN (not logged in)",
        "likely_products": likely_products_text
    })
    # clean up in case the model adds markdown formatting despite instructions
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
        return True  # general product/inventory queries don't need scoping

    if customer_id is None:
        # question touches customer-linked data but nobody is authenticated - refuse
        return False

    if str(customer_id) not in normalized:
        return False

    # Reject any use of OR when the query touches customer-scoped tables -
    # a correctly scoped single-customer query should never need to OR together
    # multiple conditions that could pull in another customer's data.
    if " or " in normalized:
        return False

    # Additional guard specifically for the customers table: even with the
    # customer's own id present, block queries that select multiple customers'
    # worth of data (e.g. no LIMIT, or selecting from customers without an
    # id/email equality filter tied to this specific customer_id).
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
    """Executes a SQL query against Postgres and returns the results.
    Refuses to run anything touching order data that isn't properly
    scoped to the authenticated customer."""
    if not is_safe_query(query):
        return None, "Query rejected: only SELECT statements are allowed."

    if not is_properly_scoped(query, customer_id):
        print(f"  [DEBUG] Query REJECTED by scoping check: {query!r}")
        return None, "Query rejected: this request requires authentication and could not be safely scoped to your account."

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


import re

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


def answer_sql_question(question: str, customer_id: int | None = None) -> str:
    """Full pipeline: question -> generated SQL -> executed -> plain-language answer.
    customer_id should be the authenticated user's id, or None if not logged in."""
    if has_destructive_intent(question):
        return ("I'm a read-only assistant and can't delete, cancel, or modify orders. "
                "Please contact customer support directly for that request.")

    query = generate_sql(question, customer_id)
    result, error = execute_sql(query, customer_id)

    if error:
        return f"Sorry, I couldn't retrieve that information. ({error})"

    if not result["rows"]:
        return "No matching records were found."

    # format results as plain text for the LLM to summarize
    formatted_rows = "\n".join(str(dict(zip(result["columns"], row))) for row in result["rows"])

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

    answer = summarize_chain.invoke({
        "question": question,
        "results": formatted_rows,
        "customer_id": customer_id if customer_id is not None else "N/A"
    })
    return answer


if __name__ == "__main__":
    test_cases = [
        ("How many AeroBook Pro 14 laptops are in stock?", None),  # general - no auth needed
        ("What is the status of my last order?", 1),                # scoped to customer 1 (Sarah)
        ("What is the status of my last order?", None),              # same question, but NOT logged in - should refuse
    ]

    for q, cid in test_cases:
        print(f"\nQ: {q} (customer_id={cid})")
        sql = generate_sql(q, cid)
        print(f"Generated SQL: {sql}")
        print(f"A: {answer_sql_question(q, cid)}")