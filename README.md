# Grocery Ecommerce RAG Assistant

An agentic customer-support backend for a grocery ecommerce store, built with
**FastAPI** and **LangGraph**. A single flattened ReAct agent handles every
customer question directly, choosing among two families of tools:

- **Structured store-data tools** (`tools.py`) — dedicated, pre-vetted
  queries for products, stock, prices, orders, cart, and reviews against the
  backend SQL Server, with `user_order_lookup`/`general_sql_lookup` as
  last-resort LLM-generated-SQL fallbacks (`text_to_sql.py`).
- **Adaptive Corrective RAG pipeline** — answers policy/FAQ questions
  (returns, shipping, payments, delivery, general product info) from a
  ChromaDB vector store of the store's documentation.

The agent keeps per-user conversation state via a Postgres-backed LangGraph
checkpointer, and automatically summarizes long conversation histories in
the background.

## Architecture

```
FastAPI (app.py)
   │
   └── ReAct agent (agent_graph.py) — Gemini 3.1 Flash Lite
        │
        ├── get_all_product_names / get_all_ordered_products_names   ─┐
        ├── get_order_by_recency, list_my_orders, get_cart_contents,  │
        │   get_saved_addresses, get_my_reviews                       ├─► tools.py ─► db.py ─► SQL Server
        ├── get_product_details, get_products_on_sale,                │
        │   get_best_selling_products, get_top_rated_products,        │
        │   get_product_reviews                                       │
        ├── user_order_lookup / general_sql_lookup (fallback) ────────┘──► text_to_sql.py
        │
        └── search_policies_and_faqs ───► A_C_rag.py (adaptive corrective RAG)
                                              │
                                              ├── classify_strategy (simple / careful)
                                              ├── fusion_retrieval_chain (RAG-Fusion, pipeline.py)
                                              ├── grade_chunks (corrective grading)
                                              └── generation_chain
                                                     │
                                                ChromaDB (chroma_data/, indexed by indexing.py)
```

### ReAct agent (`agent_graph.py`)
A single LangGraph `StateGraph` ReAct loop holding every tool directly -
the structured store-data tools (`tools.py`) and `search_policies_and_faqs`.
It decides which tool(s) to call based on the question, resolves casual
product references to exact catalog names itself before calling a detail
tool, and keeps conversation state scoped per-user (`user-{user_id}` thread
id) via `AsyncPostgresSaver`. Long conversations are summarized in the
background after each turn (`summarize_chat`) once the token count crosses
a threshold.

This used to be two agents - this outer one delegating SQL questions to a
separate `sql_ReAct.py` sub-agent via a `sql_agent_tool` wrapper. That sub-agent
was removed and its tools folded directly into this agent's tool list, cutting
one full LLM round trip off every SQL question. Tool behavior itself
(pagination, resolve-before-lookup, dedicated-tool-first) is unchanged - only
where the reasoning happens moved.

### Structured store-data tools (`tools.py`, `text_to_sql.py`)
Dedicated, pre-vetted queries for the most common question shapes (one order
by recency, cart contents, saved addresses, product details, best sellers,
top rated, reviews, ...), each running fixed SQL instead of asking an LLM to
write it. `user_order_lookup` and `general_sql_lookup` are last-resort
fallbacks that generate SQL on the fly against the schema described in
`text_to_sql.py`, for question shapes none of the dedicated tools cover.
Store-wide list results are paginated 50 rows at a time.

### Adaptive Corrective RAG (`A_C_rag.py`, `pipeline.py`, `indexing.py`)
- **Adaptive routing**: classifies each question as `simple` (single direct
  lookup → 1 retrieval + 1 generation call) or `careful` (policy edge cases,
  eligibility rules → full RAG-Fusion pipeline).
- **RAG-Fusion**: generates 4 query variants, retrieves for each, and merges
  results with Reciprocal Rank Fusion.
- **Corrective grading**: batch-grades retrieved chunks for relevance in a
  single LLM call; if nothing is relevant, rewrites the query and retries
  once before giving up.
- Documents live in `docs/` (FAQs, payments/vouchers, returns/refunds,
  shipping/delivery) and are chunked + embedded into ChromaDB by
  `indexing.py` using `sentence-transformers/all-MiniLM-L6-v2`.

### Auth (`auth.py`, `models.py`, `utils.py`, `login.py`)
JWT-based authentication scaffolding: `models.py` defines the full
SQLAlchemy schema (User, Product, Orders, Cart, Reviews, Vouchers, etc.) and
`create_all` entrypoint, `utils.py`/`auth.py` handle password hashing and
token issuing/verification, and `login.py` is a login router (note: it uses
package-relative imports and is not currently wired into `app.py`'s routes).

## Tech stack

| Layer            | Choice |
|-------------------|--------|
| API framework     | FastAPI |
| Agent orchestration | LangGraph (`StateGraph`, `ToolNode`, `AsyncPostgresSaver`) |
| Agent LLM  | Groq `openai/gpt-oss-120b` |
| RAG routing/grading/rewrite LLM | Groq `openai/gpt-oss-20b` |
| RAG classification LLM | DeepSeek `deepseek-v4-flash` |
| Embeddings        | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector store       | ChromaDB (local, `chroma_data/`) |
| Relational DB      | PostgreSQL (SQLAlchemy models + psycopg2) |
| Conversation state | Postgres via `langgraph.checkpoint.postgres.aio.AsyncPostgresSaver` |
| Auth               | JWT (`python-jose`) + `passlib` (bcrypt) |

## Project structure

```
app.py            FastAPI app: /chat and /terminate endpoints, lifespan-managed checkpointer
agent_graph.py     Single flattened ReAct agent (all tools) + conversation summarization
tools.py           Structured store-data tools + search_policies_and_faqs, exposed to the agent
text_to_sql.py      Schema description + SQL generation/execution for user & general queries
A_C_rag.py          Adaptive corrective RAG orchestration (routing, grading, rewrite)
pipeline.py          RAG-Fusion retrieval chain + generation chain, Chroma retriever
indexing.py          One-off script: chunk docs/ and (re)build the Chroma collection
docs/                Source policy/FAQ documents that get indexed
chroma_data/          Persisted Chroma vector store (generated, gitignored)
models.py            SQLAlchemy models for the Postgres schema + get_db dependency
schemas.py           Pydantic request/response models for the API
auth.py              JWT token creation/verification, get_current_user dependency
utils.py             Password hashing helpers
login.py             Login router (WIP — not yet mounted on app.py)
config.py            Pydantic settings (JWT config) loaded from .env
requirements.txt      Python dependencies
```

## Setup

### 1. Install dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure environment
Create a `.env` file in the project root with:

```
DATABASE_HOSTNAME=
DATABASE_PORT=
DATABASE_NAME=
DATABASE_USERNAME=
DATABASE_PASSWORD=

GROQ_API_KEY=
DEEPSEEK_API_KEY=

JWT_SECRET_KEY=
JWT_ALGORITHM=
JWT_EXPIRE_MINUTES=

# Optional - LangSmith tracing
LANGSMITH_TRACING=
LANGSMITH_ENDPOINT=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
```

LangGraph uses `CHECKPOINT_DB_URL` when set, otherwise it falls back to
`DATABASE_*`. The application's relational schema still uses `DATABASE_*`.

On EC2, the deployed app uses `chatbot-postgres:5432/ecommerce_rag` on its
Docker network. PostgreSQL stores its data in a Docker volume on that server.
For local development against those same checkpoint tables, configure
`EC2_HOST`, `EC2_USER`, and `EC2_KEY_PATH` in `.env`, and set
`CHECKPOINT_DB_URL=postgresql://USER:PASSWORD@127.0.0.1:15432/ecommerce_rag`
using the EC2 PostgreSQL credentials (URL-encode the username and password).
Start the tunnel in a separate terminal before starting the app:

```powershell
.\venv\Scripts\python.exe checkpoint_tunnel.py
```

Keep that terminal running. `CHECKPOINT_TUNNEL_PORT` optionally overrides
15432; update the URL to match. The tunnel uses the existing SSH known-host
entry and does not expose PostgreSQL publicly. When deploying to EC2, omit
the local `CHECKPOINT_DB_URL` override and use the container's `DATABASE_*`
settings. Do not copy the local `.env` to the server.

Local and deployed apps now share conversation state for matching user IDs.
Calling `/terminate` deletes that user's checkpoints from the shared database.

### 3. Create the database schema
```bash
python models.py
```

### 4. Index the policy/FAQ documents into ChromaDB
```bash
python indexing.py
```

### 5. Run the API
```bash
uvicorn app:app --reload
```

## API

- `GET /` — health check.
- `POST /chat` — send a question for a given user.
  ```json
  { "question": "What is your return policy?", "user_id": "1" }
  ```
  Returns `{ "question": ..., "answer": ... }`. Conversation state is kept
  per `user_id` across calls, and old messages are summarized in the
  background once the thread gets long.
- `POST /terminate` — deletes the stored conversation thread for a user.
  ```json
  { "user_id": "1" }
  ```
