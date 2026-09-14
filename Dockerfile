# Chatbot Assistant / Grocery RAG API — FastAPI + LangGraph.
# CPU-only inference. The embedding model and the Chroma vector store are
# baked in at build time so the container starts fast and needs no network
# for the RAG path. Relational chat-checkpoint data lives in a sibling
# Postgres container. Product/order/account data lives in an external
# SQL Server instance, reached over ODBC.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev curl gnupg apt-transport-https unixodbc unixodbc-dev \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl -sSL https://packages.microsoft.com/config/debian/12/prod.list -o /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch (matches the visual-search sibling; keeps the image small)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1

COPY requirements.txt .
# `database==0.6.0` is a stray, unused pin with no matching distribution — drop it.
RUN grep -vE '^\s*database==' requirements.txt > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir \
        "uvicorn[standard]" \
        "psycopg[binary,pool]" \
        "langgraph-checkpoint-postgres" \
        "sentence-transformers" \
        "langchain_google_genai"

COPY . .

# Pre-download the embedding model and build the Chroma collection at build time.
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')" \
    && python indexing.py \
    && chmod +x entrypoint.sh

EXPOSE 8000

CMD ["./entrypoint.sh"]
