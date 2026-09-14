# Redeploying chatbot.54-93-65-9.sslip.io to the current main

The server is currently running an OLD version - confirmed live (2026-09-14):
asking it "do you accept vouchers?" answers yes with a promo-code field. The
current `main` has no voucher feature at all (the backend dropped it) and
reads product/order data from SQL Server instead of the old local Postgres.
This is a real content/functionality gap, not just stale code.

Run this AS the user/account that already runs the app (so file ownership and
any existing service definition stay correct). Needs SSH access to
`54.93.65.9`.

---

## 0. Find out how the app is currently being kept running

Nothing in the repo defines a systemd service, so whoever deployed this set
one up by hand (or is running it in `screen`/`tmux`/`nohup`). Find out which
before touching anything, or the restart step below won't apply:

```bash
systemctl status chatbot 2>&1 || systemctl list-units --type=service | grep -i chat
ps aux | grep -i uvicorn
```

If it's a systemd service, note its name (used as `<service>` below). If it's
a bare process, note the working directory it's running from.

---

## 1. Back up the current `.env`

The new version needs one extra variable (`CHECKPOINT_DB_URL`) that the old
one never had - don't lose what's already working.

```bash
cd /path/to/Chatbot-Assistant     # wherever it's checked out on the server
cp .env .env.backup.$(date +%Y%m%d)
```

## 2. Pull the new code

```bash
git fetch origin
git status                         # make sure nothing local is uncommitted first
git checkout main
git pull origin main
git log --oneline -1               # should show b86d4a0 or later
```

## 3. Install the system ODBC driver (new requirement)

The old version talked to a local Postgres for everything. The new version
reads products/orders/cart from the backend's SQL Server via `pyodbc`, which
needs a system-level driver `apt` must install - `pip install` alone is not
enough. Skip this and every product/order question will fail with
`Can't open lib 'ODBC Driver 18 for SQL Server'`.

```bash
curl -sSL https://packages.microsoft.com/keys/microsoft.asc \
  | sudo gpg --dearmor -o /usr/share/keyrings/microsoft.gpg
echo "deb [arch=amd64,signed-by=/usr/share/keyrings/microsoft.gpg] \
  https://packages.microsoft.com/ubuntu/22.04/prod jammy main" \
  | sudo tee /etc/apt/sources.list.d/mssql-release.list
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get install -y msodbcsql18 unixodbc
odbcinst -q -d              # confirm it lists [ODBC Driver 18 for SQL Server]
```

## 4. Update Python dependencies

`requirements.txt` changed significantly (added `pyodbc`, `psycopg`,
`psycopg-pool`, `langgraph-checkpoint-postgres`, `sentence-transformers`,
`chromadb`, `langchain-google-genai`; removed several unused ones).

```bash
source venv/bin/activate            # or wherever the venv lives
pip install -r requirements.txt
```

## 5. Keep persistence on this EC2 server

Use the persistence override described in DEPLOYMENT.md. The server already
has PostgreSQL and existing checkpoint data in its Docker volume. Do not add
a Neon URL. Preserve the server's DATABASE_* credentials and apply the
CHECKPOINT_DB_* settings from compose.persistence.yml.

The current source also needs MSSQL_* and model-provider settings from .env.
Deploying this source requires updating the existing Docker image and its
system dependencies; changing persistence settings alone does not update code.

## 6. Rebuild the vector store

This is the step that fixes the wrong voucher answer. `chroma_data/` is
gitignored, so the server is still serving the OLD FAQ content baked into its
existing index - pulling the code alone does not update it.

```bash
rm -rf chroma_data
python setup_checkpoint_db.py       # verify PostgreSQL before restarting the app
python indexing.py                  # rebuilds the vector store from docs/*.txt
```

`setup_checkpoint_db.py` should end with `[ok] chat persistence is ready.`
`indexing.py` should end with `Indexing complete.` and mention 6 documents.

## 7. Restart the app

Whichever applies, from step 0:

```bash
sudo systemctl restart <service>
# or, if it's a bare process: kill the old uvicorn PID, then start it the
# same way it was started before (check for a start script or systemd
# override) - do not guess a new invocation.
```

## 8. Verify from outside

```bash
curl -X POST https://chatbot.54-93-65-9.sslip.io/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"do you accept vouchers?","user_id":"redeploy-check"}'
```

Expect something like *"there are no voucher, promo or refund codes"* - NOT
the old "yes, enter your promo code" answer. If it still says yes, the vector
store rebuild in step 6 either didn't run or didn't get picked up by the
restarted process (double-check `chroma_data/` is actually the rebuilt one and
the restart in step 7 actually happened).

Then confirm SQL Server and persistence:

```bash
curl -X POST https://chatbot.54-93-65-9.sslip.io/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"how much does ginger cost?","user_id":"redeploy-check"}'

curl -X POST https://chatbot.54-93-65-9.sslip.io/terminate \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"redeploy-check"}'
```

Check the app's own startup log for `[startup] chat persistence -> ...` - it
must show `chatbot-postgres:5432/ecommerce_rag`.
