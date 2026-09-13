# Deployment checklist (EC2)

Things that are NOT handled by `git push` + `pip install`. Each of these
will stop the app or silently break a feature if skipped.

---

## 1. System-level ODBC driver (blocks all store/product questions)

`pyodbc` is a Python binding, not the driver itself. Without the system driver
every `sql_agent_tool` call fails with
`Can't open lib 'ODBC Driver 18 for SQL Server' : file not found`.

Amazon Linux 2023:

```bash
curl https://packages.microsoft.com/config/rhel/9/prod.repo \
  | sudo tee /etc/yum.repos.d/mssql-release.repo
sudo ACCEPT_EULA=Y dnf install -y msodbcsql18 unixODBC
odbcinst -q -d            # should list [ODBC Driver 18 for SQL Server]
```

Ubuntu 22.04:

```bash
curl -sSL https://packages.microsoft.com/keys/microsoft.asc \
  | sudo gpg --dearmor -o /usr/share/keyrings/microsoft.gpg
echo "deb [arch=amd64,signed-by=/usr/share/keyrings/microsoft.gpg] \
  https://packages.microsoft.com/ubuntu/22.04/prod jammy main" \
  | sudo tee /etc/apt/sources.list.d/mssql-release.list
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get install -y msodbcsql18 unixodbc
```

---

## 2. Build the vector store (blocks all policy/FAQ answers)

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

## 3. Outbound network the instance must be able to reach

The app's security group needs egress to all of these:

| Destination | Port | Used for |
|---|---|---|
| RDS endpoint (private, same VPC) | 5432 | chat persistence |
| `63.183.213.35` (backend SQL Server) | 1433 | products, orders, cart |
| `generativelanguage.googleapis.com` | 443 | Gemini |
| `api.deepseek.com` | 443 | DeepSeek (SQL fallbacks) |
| `huggingface.co` | 443 | embedding model download (first run) |

**Check with the backend team whether their SQL Server firewall allows the
EC2 instance's IP.** It is a remote host outside our VPC; if it only
whitelists known addresses, the new instance will be refused. This is the
single most likely thing to work locally and fail on EC2.

Verify from the instance:

```bash
nc -vz 63.183.213.35 1433      # backend SQL Server
nc -vz <rds-endpoint> 5432     # chat persistence
```

---

## 4. Environment variables

`.env` is gitignored (correctly - it holds the SQL Server password and API
keys), so it does **not** arrive with the code. Supply the variables on the
instance, ideally from SSM Parameter Store or Secrets Manager rather than a
file on disk. See `.env.example` for the full list.

Minimum for the app to start:

```
CHECKPOINT_DB_URL=postgresql://user:pass@<rds-endpoint>:5432/ragchat
MSSQL_SERVER=63.183.213.35,1433
MSSQL_DATABASE=EcommerceDB
MSSQL_USER=...
MSSQL_PASSWORD=...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...
```

`CHECKPOINT_DB_URL` must point at RDS. If it is unset and the legacy
`DATABASE_HOSTNAME=localhost` is used instead, the app will try to reach a
Postgres on the EC2 instance itself and fail at startup.

---

## 5. First-boot verification

The checkpoint tables are created automatically by
`AsyncPostgresSaver.setup()` - no migrations to run. Confirm on startup:

```
[startup] chat persistence -> <rds-endpoint>:5432/ragchat
```

If that line says `localhost`, the environment is still pointing at the
wrong database.

Then smoke-test both paths and persistence:

```bash
curl -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"what is the return policy?","user_id":"1"}'   # policy path

curl -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"how much does ginger cost?","user_id":"1"}'   # SQL path

curl -X POST localhost:8000/terminate \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"1"}'
```

Restart the app and ask a follow-up question as the same `user_id`: if the
assistant still has the earlier context, persistence is working against RDS.

---

## 6. Known dead files

`auth.py` and `config.py` are not imported by the running app (the live
routes are `/`, `/chat`, `/terminate`, `/test`). `auth.py` imports
`models.py`, which has been deleted, so it would fail if anyone wired it up.
Either delete it or restore `models.py` before using it.
