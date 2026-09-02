# Grocery Ecommerce RAG Assistant

An agentic customer-support backend for a grocery ecommerce store, built with
**FastAPI** and **LangGraph**. A ReAct-style orchestrator agent routes each
customer question to one of two specialized sub-systems:

- **Text-to-SQL agent** — answers questions about products, stock, prices,
  orders, cart, reviews, and vouchers by generating and executing SQL against
  the store's PostgreSQL database.
- **Adaptive Corrective RAG pipeline** — answers policy/FAQ questions
  (returns, shipping, payments, delivery, general product info) from a
  ChromaDB vector store of the store's documentation.

The orchestrator keeps per-user conversation state via a Postgres-backed
LangGraph checkpointer, and automatically summarizes long conversation
histories in the background.

## Architecture

```
FastAPI (app.py)
   │
   └── ReAct orchestrator agent (agent_graph.py)  — Groq openai/gpt-oss-120b
        │
        ├── sql_agent_tool ─────────────► sql_ReAct.py sub-agent — DeepSeek
        │                                    │  (tools.py)
        │                                    ├── get_all_product_names
        │                                    ├── get_all_ordered_products_names
        │                                    ├── user_order_lookup ────► text_to_sql.py
        │                                    └── general_sql_lookup ───► text_to_sql.py
        │                                                                    │
        │                                                              PostgreSQL (models.py)
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

### Orchestrator agent (`agent_graph.py`)
A LangGraph `StateGraph` ReAct loop with exactly two tools: `sql_agent_tool`
and `search_policies_and_faqs`. It decides which tool(s) to call based on the
question, combines results when a question spans both domains, and keeps
conversation state scoped per-user (`user-{user_id}` thread id) via
`AsyncPostgresSaver`. Long conversations are summarized in the background
after each turn (`summarize_chat`) once the token count crosses a threshold.

### SQL sub-agent (`sql_ReAct.py`, `tools.py`, `text_to_sql.py`)
A separate, independent ReAct agent (deliberately isolated from the RAG side)
that resolves casual product references to exact catalog names, then answers
either user-scoped questions (`user_order_lookup` — orders, cart, addresses,
own reviews) or store-wide questions (`general_sql_lookup` — catalog,
aggregate reviews, vouchers) by generating SQL against the schema described
in `text_to_sql.py`. Store-wide list results are paginated 50 rows at a time.

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
| Orchestrator LLM  | Groq `openai/gpt-oss-120b` |
| SQL sub-agent LLM | DeepSeek `deepseek-v4-flash` |
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
agent_graph.py     Top-level ReAct orchestrator agent + conversation summarization
tools.py           Tools exposed to the orchestrator and SQL sub-agent
sql_ReAct.py        SQL sub-agent (ReAct loop over SQL tools)
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

The same `DATABASE_*` variables are used both for the application's own
relational schema (`models.py`) and for the LangGraph Postgres checkpointer
(`agent_graph.py`), so they must point at a database the app can create
tables in.

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
