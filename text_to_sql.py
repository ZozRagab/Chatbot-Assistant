import os
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

customers(id, name, email)
    - one row per customer

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

Question: {question}

Respond with ONLY the raw SQL query. No explanation, no markdown formatting, no backticks."""

sql_prompt = ChatPromptTemplate.from_template(sql_template)

sql_generation_chain = sql_prompt | llm | StrOutputParser()


def generate_sql(question: str) -> str:
    """Generates a SQL query string from a natural language question."""
    raw_output = sql_generation_chain.invoke({
        "schema": SCHEMA_DESCRIPTION,
        "question": question
    })
    # clean up in case the model adds markdown formatting despite instructions
    cleaned = raw_output.strip().strip("`").replace("sql\n", "", 1).strip()
    return cleaned


def is_safe_query(query: str) -> bool:
    """Basic safety check - only allow SELECT statements."""
    normalized = query.strip().lower()
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "create"]
    if not normalized.startswith("select"):
        return False
    if any(word in normalized for word in forbidden):
        return False
    return True


def execute_sql(query: str):
    """Executes a SQL query against Postgres and returns the results."""
    if not is_safe_query(query):
        return None, "Query rejected: only SELECT statements are allowed."

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


DESTRUCTIVE_INTENT_KEYWORDS = ["delete", "remove", "cancel my order", "update", "change my", "modify"]


def has_destructive_intent(question: str) -> bool:
    """Checks if the user's question is asking for a write/delete action,
    not just an information lookup. This must run BEFORE SQL generation -
    checking only the generated SQL's syntax isn't enough, since the LLM
    can (correctly) generate a safe SELECT while the final natural-language
    answer still ends up phrased as if it performed the requested action."""
    normalized = question.lower()
    return any(word in normalized for word in DESTRUCTIVE_INTENT_KEYWORDS)


def answer_sql_question(question: str) -> str:
    """Full pipeline: question -> generated SQL -> executed -> plain-language answer."""
    if has_destructive_intent(question):
        return ("I'm a read-only assistant and can't delete, cancel, or modify orders. "
                "Please contact customer support directly for that request.")

    query = generate_sql(question)
    result, error = execute_sql(query)

    if error:
        return f"Sorry, I couldn't retrieve that information. ({error})"

    if not result["rows"]:
        return "No matching records were found."

    # format results as plain text for the LLM to summarize
    formatted_rows = "\n".join(str(dict(zip(result["columns"], row))) for row in result["rows"])

    summarize_template = """Given this question and the raw database results below,
answer the question in a natural, friendly sentence.

Question: {question}

Database results:
{results}

Answer:"""
    summarize_prompt = ChatPromptTemplate.from_template(summarize_template)
    summarize_chain = summarize_prompt | llm | StrOutputParser()

    answer = summarize_chain.invoke({"question": question, "results": formatted_rows})
    return answer


if __name__ == "__main__":
    test_questions = [
        "How many AeroBook Pro 14 laptops are in stock?",
        "What is the status of order 3?",
        "What did customer Sarah Ahmed order?",
        "Delete Sarah Ahmed last order"
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        sql = generate_sql(q)
        print(f"Generated SQL: {sql}")
        print(f"A: {answer_sql_question(q)}")