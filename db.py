"""SQL Server connection for the store's business data (products, orders, cart,
reviews). Postgres is still used, but ONLY for LangGraph chat checkpoints - see
DB_URI in agent_graph.py / app.py.

Shared by tools.py and text_to_sql.py. It lives in its own module because
tools.py imports text_to_sql, so neither of those can own it without a
circular import.

WHY A POOL
The server is remote (an internet hop from wherever this runs), so opening a
connection costs TCP + TLS + login: ~0.25s typically, ~0.9s the first time. The
queries themselves take ~0.1s, of which ~0.11s is the network round trip - i.e.
the database does its work in a few ms. Measured: open+query+close per call was
~0.5s; the same query on an already-open connection is ~0.095s. pyodbc's own
`pooling` flag did not help here (connect+close still swung 0.17s-2.09s), so
connections are kept open and reused explicitly below.

RULES THE POOL ENFORCES
- A connection is checked out by exactly one caller at a time. SQL Server has no
  MARS by default: a second query on a connection whose previous result set is
  still pending raises "Connection is busy with results for another command".
  Checkout gives each caller exclusive use, so that cannot happen across tools.
  Within one checkout, still run queries one at a time and read each fully.
- A connection that raised is discarded, not returned - it may be half-dead.
- A connection that has sat idle for a while is pinged before reuse and
  replaced if the server dropped it (SQL Server closes idle sessions). The
  ping is NOT done on every checkout: it is itself a network round trip
  (~0.11s), which would eat most of the saving from pooling in the first
  place. A connection used within the last PING_AFTER_IDLE_S seconds is
  handed out as-is; if it turns out to be dead the query raises, the
  connection is discarded, and the caller sees the error once.
"""
import os
import queue
import threading
import time
from contextlib import contextmanager

import pyodbc
from dotenv import load_dotenv

load_dotenv()

# How many connections to keep open at most. Every tool call holds one only for
# the duration of its query, so this is the number of *concurrent* DB-touching
# requests we expect, not the number of users.
POOL_SIZE = int(os.getenv("MSSQL_POOL_SIZE", "5"))

# Only health-check a pooled connection if it has been idle at least this long.
PING_AFTER_IDLE_S = float(os.getenv("MSSQL_PING_AFTER_IDLE_S", "60"))


def _connect():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={os.getenv('MSSQL_SERVER')};"
        f"DATABASE={os.getenv('MSSQL_DATABASE')};"
        f"UID={os.getenv('MSSQL_USER')};"
        # braces let the password contain ';' or other ODBC-special characters
        f"PWD={{{os.getenv('MSSQL_PASSWORD')}}};"
        "Encrypt=yes;TrustServerCertificate=yes;",
        timeout=15,
        # read-only workload: no transactions to manage, and it avoids holding
        # an implicit transaction open on a pooled connection between checkouts
        autocommit=True,
    )


def get_connection():
    """A NEW, unpooled connection. Kept for callers that explicitly want their
    own (e.g. schema introspection scripts). Application code should use
    `connection()` / `fetch_all()` so it goes through the pool."""
    return _connect()


class _Pool:
    def __init__(self, size: int):
        # Each idle entry is (connection, time it was last released).
        self._idle: "queue.LifoQueue[tuple[pyodbc.Connection, float]]" = queue.LifoQueue()
        self._size = size
        self._created = 0
        self._lock = threading.Lock()

    def _is_alive(self, conn) -> bool:
        try:
            cur = conn.cursor()
            try:
                cur.execute("SELECT 1").fetchall()
            finally:
                cur.close()
            return True
        except pyodbc.Error:
            return False

    def _discard(self, conn) -> None:
        try:
            conn.close()
        except pyodbc.Error:
            pass
        with self._lock:
            self._created -= 1

    def acquire(self):
        # Prefer an idle connection (LIFO: the most recently used one is the
        # least likely to have been dropped by the server).
        while True:
            try:
                conn, released_at = self._idle.get_nowait()
            except queue.Empty:
                break
            recently_used = (time.monotonic() - released_at) < PING_AFTER_IDLE_S
            if recently_used or self._is_alive(conn):
                return conn
            self._discard(conn)

        # None idle - open a new one if we're under the cap...
        with self._lock:
            can_create = self._created < self._size
            if can_create:
                self._created += 1
        if can_create:
            try:
                return _connect()
            except Exception:
                with self._lock:
                    self._created -= 1
                raise

        # ...otherwise wait for one to be returned. (Just released, so no ping.)
        conn, _released_at = self._idle.get()
        return conn

    def release(self, conn, *, broken: bool = False) -> None:
        if broken:
            self._discard(conn)
        else:
            self._idle.put((conn, time.monotonic()))

    def warm(self, n: int = 1) -> None:
        """Open `n` connections up front so the first request doesn't pay the
        ~0.9s cold handshake. Called from app startup."""
        conns = [self.acquire() for _ in range(min(n, self._size))]
        for c in conns:
            self.release(c)


_pool = _Pool(POOL_SIZE)


def warm_pool(n: int = 2) -> None:
    _pool.warm(n)


@contextmanager
def connection():
    """Check a pooled connection out for exclusive use, then hand it back.

    Use this when a tool needs several sequential queries (see
    get_order_by_recency). Run them one at a time and read each result fully
    before the next - see the module docstring on MARS.
    """
    conn = _pool.acquire()
    broken = False
    try:
        yield conn
    except pyodbc.Error:
        # The connection may be in an unusable state (mid-result-set, dropped
        # socket, ...). Don't put it back for the next caller to trip over.
        broken = True
        raise
    finally:
        _pool.release(conn, broken=broken)


def fetch_on(conn, sql: str, params: tuple = ()) -> list:
    """Run a query on a checked-out connection (see `connection`)."""
    cursor = conn.cursor()
    try:
        cursor.execute(sql, params)
        return cursor.fetchall()
    finally:
        # Free the result set so the connection is reusable immediately.
        cursor.close()


def fetch_all(sql: str, params: tuple = ()) -> list:
    """Run one read-only query on a pooled connection and return all rows."""
    with connection() as conn:
        return fetch_on(conn, sql, params)


def placeholders(values) -> str:
    """'?, ?, ?' for an IN (...) clause. T-SQL has no array parameter like
    Postgres's = ANY(%s), so list filters expand to one '?' per value."""
    return ", ".join("?" for _ in values)
