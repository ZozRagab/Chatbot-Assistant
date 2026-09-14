# Deployment checklist

Things that are NOT handled by `git push` + `pip install`. Each of these will
stop the app or silently break a feature if skipped.

---

## 1. Chat persistence on the AWS app server

The existing EC2 deployment runs PostgreSQL 16 in `chatbot-postgres`, beside
`chatbot-assistant`. Data is stored in the Docker volume
`chatbot-assistant_chatbot-postgres-data`. Port 5432 is not published.
Both containers restart automatically unless explicitly stopped.

Use the existing server directory `/opt/ai-services/chatbot-assistant`.
The checked-in `compose.persistence.yml` records the persistence settings.
Install its contents as `docker-compose.override.yml` in that directory so
ordinary `docker compose up -d` also applies them. If an override already exists,
merge the settings instead of overwriting it.

The override clears `CHECKPOINT_DB_URL` (which otherwise wins over all other
settings), uses `chatbot-postgres:5432`, and reuses the existing database
credentials from the server's `.env`. TLS is disabled only on the private
Docker network. `localhost` inside the app container would refer to the app
container itself, not the PostgreSQL container.

Run `python setup_checkpoint_db.py` inside the app container after deploying
the current source. It idempotently creates/verifies `checkpoints`,
`checkpoint_blobs`, `checkpoint_writes`, and `checkpoint_migrations`.
Keep the connection pool: it recovers from stale connections after DB restarts.

Never run `docker compose down -v`: that removes the persistent database volume.
Back up with `pg_dump` before migration. Neon and AWS may contain different chat
histories; do not restore one over the other without a deliberate migration.

---

## 2. System-level ODBC driver (blocks all store/product questions)

`pyodbc` is a Python binding, not the driver itself. Without the system driver
every `sql_agent_tool` call fails with
`Can't open lib 'ODBC Driver 18 for SQL Server' : file not found`.

Ubuntu 22.04 (matches the current server):

```bash
curl -sSL https://packages.microsoft.com/keys/microsoft.asc \
  | sudo gpg --dearmor -o /usr/share/keyrings/microsoft.gpg
echo "deb [arch=amd64,signed-by=/usr/share/keyrings/microsoft.gpg] \
  https://packages.microsoft.com/ubuntu/22.04/prod jammy main" \
  | sudo tee /etc/apt/sources.list.d/mssql-release.list
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get install -y msodbcsql18 unixodbc
odbcinst -q -d            # should list [ODBC Driver 18 for SQL Server]
```

Amazon Linux 2023:

```bash
curl https://packages.microsoft.com/config/rhel/9/prod.repo \
  | sudo tee /etc/yum.repos.d/mssql-release.repo
sudo ACCEPT_EULA=Y dnf install -y msodbcsql18 unixODBC
```

---

## 3. Build the vector store (blocks all policy/FAQ answers)

`chroma_data/` is gitignored, so a fresh checkout has **no vector store**.
The `docs/*.txt` sources and `indexing.py` ARE in git - the index just has to
be built on the instance, once, before first start and again whenever
`docs/` changes:

```bash
python indexing.py     # "Loaded 6 documents / Split into 35 chunks"
```

First run also downloads the `all-MiniLM-L6-v2` embedding model from
HuggingFace (~90 MB), so the instance needs outbound internet for that.

---

## 4. Outbound network the instance must be able to reach

| Destination | Port | Used for |
|---|---|---|
| `63.183.213.35` (backend SQL Server) | 1433 | products, orders, cart |
| `generativelanguage.googleapis.com` | 443 | Gemini |
| `api.deepseek.com` | 443 | DeepSeek (SQL fallbacks) |
| `huggingface.co` | 443 | embedding model download (first run) |

**Check with the backend team whether their SQL Server firewall allows the
server's IP.** It is a remote host outside our network; if it only whitelists
known addresses, a new instance will be refused. This is the most likely thing
to work locally and fail in deployment.

Verify from the instance:

```bash
nc -vz 63.183.213.35 1433
```

---

## 5. Environment variables

`.env` is gitignored (correctly - it holds the SQL Server password and API
keys), so it does **not** arrive with the code. See `.env.example` for the
full list. Minimum for the app to start:

```
CHECKPOINT_DB_HOST=chatbot-postgres
CHECKPOINT_DB_SSLMODE=disable
# Remaining CHECKPOINT_DB_* values are supplied by the Compose override.
MSSQL_SERVER=63.183.213.35,1433
MSSQL_DATABASE=EcommerceDB
MSSQL_USER=...
MSSQL_PASSWORD=...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...
```

If `CHECKPOINT_DB_URL` is unset, the app falls back to the legacy
`DATABASE_HOSTNAME` (localhost) and will fail on a server that has no local
Postgres.

---

## 6. First-boot verification

Startup prints the persistence target:

```
[startup] chat persistence -> chatbot-postgres:5432/ecommerce_rag
```

For this Docker deployment the target must be `chatbot-postgres`, not Neon or
`localhost`.

Then smoke-test both answer paths and persistence:

```bash
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"question":"what is the return policy?","user_id":"1"}'   # policy path

curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"question":"how much does ginger cost?","user_id":"1"}'   # SQL path

curl -X POST localhost:8000/terminate -H 'Content-Type: application/json' \
  -d '{"user_id":"1"}'
```

Restart the app and ask a follow-up as the same `user_id`: if the assistant
still has the earlier context, persistence is working.

---

## 7. Known dead files

`auth.py` and `config.py` are not imported by the running app (the live routes
are `/`, `/chat`, `/terminate`, `/test`). `auth.py` imports `models.py`, which
has been deleted, so it would fail if anyone wired it up. Either delete it or
restore `models.py` before using it.
