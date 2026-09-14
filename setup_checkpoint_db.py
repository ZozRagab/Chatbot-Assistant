"""Create the chat-persistence tables on a fresh database (Neon, RDS, local).

Standalone so the schema can be created and verified without booting the whole
app - no model providers, no vector store, no SQL Server. Reads the connection
details from the environment (see checkpoint_db.py), so no credential is ever
passed on the command line.

    python setup_checkpoint_db.py

Safe to re-run: LangGraph's setup() is idempotent and leaves existing rows
alone, so this can be used to verify an already-configured database too.
"""
import asyncio
import sys

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from checkpoint_db import describe_target, get_checkpoint_db_url

EXPECTED_TABLES = [
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
]


async def main() -> int:
    try:
        url = get_checkpoint_db_url()
    except RuntimeError as e:
        print(f"[error] {e}")
        return 1

    print(f"[setup] target: {describe_target()}")

    pool = AsyncConnectionPool(
        conninfo=url,
        min_size=1,
        max_size=2,
        open=False,
        check=AsyncConnectionPool.check_connection,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )

    try:
        await pool.open(wait=True)
    except Exception as e:
        print(f"[error] could not connect: {type(e).__name__}: {e}")
        print("        check CHECKPOINT_DB_URL in .env - host, password, and "
              "that the database accepts connections from this machine.")
        return 1

    try:
        async with pool.connection() as conn:
            row = await (await conn.execute("SELECT version()")).fetchone()
            print(f"[setup] server: {str(row['version']).split(',')[0]}")

        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        print("[setup] checkpointer.setup() completed")

        async with pool.connection() as conn:
            rows = await (await conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            )).fetchall()
        found = [r["table_name"] for r in rows]

        print(f"[setup] tables now present: {found}")
        missing = [t for t in EXPECTED_TABLES if t not in found]
        if missing:
            print(f"[error] expected tables missing: {missing}")
            return 1

        print("[ok] chat persistence is ready.")
        return 0
    finally:
        await pool.close()


if __name__ == "__main__":
    if sys.platform == "win32":
        # psycopg async needs a SelectorEventLoop; Windows defaults to Proactor.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main()))
