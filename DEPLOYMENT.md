# Deployment checklist

Things that are NOT handled by `git push` + `pip install`. Each of these will
stop the app or silently break a feature if skipped.

---

## 1. Chat-persistence database (Neon)

Conversation history lives in PostgreSQL, hosted on [Neon](https://neon.tech)
- serverless Postgres, free tier, no credit card, no VPC or networking setup.

**One-time setup:**

1. Create a Neon account and a project (any region close to where the app
   runs).
2. Copy the connection string Neon shows you. It looks like:
   `postgresql://USER:PASSWORD@ep-xxx-xxx.REGION.aws.neon.tech/neondb?sslmode=require`
3. Put it in `.env` (and in whatever the deployed instance uses for
   environment variables):

   ```
   CHECKPOINT_DB_URL=postgresql://USER:PASSWORD@ep-xxx.REGION.aws.neon.tech/neondb?sslmode=require
   ```

4. Create the tables:

   ```bash
   python setup_checkpoint_db.py
   ```

   Expected output ends with `[ok] chat persistence is ready.` and lists the
   four tables: `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`,
   `checkpoint_migrations`.

That is the whole database setup. The app also calls `setup()` itself on
startup, so this script is really a way to verify the connection before
deploying.

**Why the app uses a connection pool:** Neon suspends an idle database after a
few minutes. A single long-lived connection held across that suspend comes
back dead. `app.py` uses `AsyncConnectionPool` with a `check` callback, which
validates a connection before use and silently replaces a stale one. Do not
"simplify" this back to a single connection.

**Free-tier note:** a suspended database takes roughly half a second to wake
on the first query. Subsequent requests are normal speed.

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
| `*.neon.tech` | 5432 | chat persistence |
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
CHECKPOINT_DB_URL=postgresql://...neon.tech/neondb?sslmode=require
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
[startup] chat persistence -> ep-xxx.REGION.aws.neon.tech:5432/neondb
```

If that line says `localhost`, the environment is still pointing at the wrong
database.

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
