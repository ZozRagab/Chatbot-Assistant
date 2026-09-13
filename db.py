"""SQL Server connection for the store's business data (products, orders, cart,
reviews). Postgres is still used, but ONLY for LangGraph chat checkpoints - see
DB_URI in agent_graph.py / app.py.

Shared by tools.py and text_to_sql.py. It lives in its own module because
tools.py imports text_to_sql, so neither of those can own it without a
circular import.
"""
import os
from contextlib import contextmanager

import pyodbc
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={os.getenv('MSSQL_SERVER')};"
        f"DATABASE={os.getenv('MSSQL_DATABASE')};"
        f"UID={os.getenv('MSSQL_USER')};"
        # braces let the password contain ';' or other ODBC-special characters
        f"PWD={{{os.getenv('MSSQL_PASSWORD')}}};"
        "Encrypt=yes;TrustServerCertificate=yes;",
        timeout=15,
    )


def fetch_all(sql: str, params: tuple = ()) -> list:
    """Run one read-only query on a fresh connection and return all rows.

    One connection per call on purpose: SQL Server rejects a second query on a
    connection whose previous result set hasn't been fully read (no MARS), so
    sharing a connection across tool calls is a foot-gun.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return cursor.fetchall()
    finally:
        conn.close()


@contextmanager
def connection():
    """Share ONE connection across several sequential queries.

    The server is remote, so opening a connection costs ~0.3s (TCP + TLS +
    auth) while the query itself is ~0.08s. A tool that needs more than one
    query should use this instead of calling fetch_all twice - that would pay
    the connection cost again.

    Still one query at a time, each fully read before the next: SQL Server has
    no MARS by default and errors on a second query while results are pending.
    """
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def fetch_on(conn, sql: str, params: tuple = ()) -> list:
    """Run a query on an already-open connection (see `connection`)."""
    cursor = conn.cursor()
    cursor.execute(sql, params)
    return cursor.fetchall()


def placeholders(values) -> str:
    """'?, ?, ?' for an IN (...) clause. T-SQL has no array parameter like
    Postgres's = ANY(%s), so list filters expand to one '?' per value."""
    return ", ".join("?" for _ in values)
