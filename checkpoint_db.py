"""Connection URL for the chat-persistence database (PostgreSQL).

This database stores ONLY LangGraph's conversation checkpoints - the four
checkpoint_* tables that let a customer's chat survive a restart. It is
deliberately separate from the store's business data, which lives on the
backend team's SQL Server and is reached through db.py. Nothing here should
ever point at that server, and nothing in db.py should ever point at this one.

Deployment target: Postgres running on the SAME server as the app (installed
directly, not a managed service like RDS/Neon) - chosen to avoid the extra
network hop to an external database on every checkpoint read/write. That
means the deployed value of CHECKPOINT_DB_HOST is normally "localhost", same
as local development - there is no separate "prod" host to configure. A
managed/remote Postgres still works if the project ever needs one; nothing
below assumes local-only.

Configuration, in order of precedence:

  1. CHECKPOINT_DB_URL  - a full postgresql:// URL, for a remote host that
                          hands you one directly (e.g. a managed provider).
  2. CHECKPOINT_DB_HOST / _PORT / _NAME / _USER / _PASSWORD  - discrete parts.
  3. DATABASE_HOSTNAME / _PORT / _NAME / _USERNAME / _PASSWORD  - the legacy
     local-development names, kept so an existing .env keeps working.

TLS: defaults to sslmode=require for any non-local host, sslmode=prefer for
localhost (the normal case here, no TLS setup needed on the same box).
Override with CHECKPOINT_DB_SSLMODE.
"""
import os
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


def _sslmode_for(host: str) -> str:
    explicit = os.getenv("CHECKPOINT_DB_SSLMODE")
    if explicit:
        return explicit
    # Local dev has no TLS set up; anything remote (RDS) must be encrypted.
    return "prefer" if host in _LOCAL_HOSTS else "require"


def _from_parts() -> str | None:
    """Build a URL from discrete settings, preferring the CHECKPOINT_DB_* names
    and falling back to the legacy DATABASE_* ones."""
    host = os.getenv("CHECKPOINT_DB_HOST") or os.getenv("DATABASE_HOSTNAME")
    name = os.getenv("CHECKPOINT_DB_NAME") or os.getenv("DATABASE_NAME")
    user = os.getenv("CHECKPOINT_DB_USER") or os.getenv("DATABASE_USERNAME")
    password = os.getenv("CHECKPOINT_DB_PASSWORD") or os.getenv("DATABASE_PASSWORD")
    port = os.getenv("CHECKPOINT_DB_PORT") or os.getenv("DATABASE_PORT") or "5432"

    if not all([host, name, user, password]):
        return None

    # quote_plus: RDS master passwords routinely contain @ : / # and friends,
    # any of which would otherwise split the URL in the wrong place.
    return (
        f"postgresql://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{name}?sslmode={_sslmode_for(host)}"
    )


def get_checkpoint_db_url() -> str:
    """The URL AsyncPostgresSaver connects to. Raises if unconfigured rather
    than letting the app start and fail on the first customer request."""
    url = os.getenv("CHECKPOINT_DB_URL") or _from_parts()
    if not url:
        raise RuntimeError(
            "Chat-persistence database is not configured. Set CHECKPOINT_DB_URL "
            "to PostgreSQL, or CHECKPOINT_DB_HOST/_PORT/_NAME/_USER/"
            "_PASSWORD. (This is the PostgreSQL checkpoint store - not the "
            "backend SQL Server, which is configured separately via MSSQL_*.)"
        )
    return url


def describe_target() -> str:
    """Host/database only - safe to log, never includes the password."""
    url = get_checkpoint_db_url()
    try:
        after_at = url.split("@", 1)[1]
        hostport, _, dbpart = after_at.partition("/")
        return f"{hostport}/{dbpart.split('?', 1)[0]}"
    except (IndexError, ValueError):
        return "(unparsed)"


# Importable constant for existing call sites.
DB_URI = get_checkpoint_db_url()
