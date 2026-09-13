import asyncio
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from session_insights import get_suggested_product_ids
from schemas import QuestionRequest, AnswerResponse, TerminationRequest, TerminationResponse
from agent_graph import graph, summarize_chat
from checkpoint_db import describe_target, get_checkpoint_db_url
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# psycopg's async mode requires a SelectorEventLoop, but Windows defaults the
# main thread to ProactorEventLoop - which raises psycopg.InterfaceError the
# moment AsyncPostgresSaver opens its connection in lifespan(), below. Must
# be set before uvicorn/asyncio picks a loop, so this runs at import time.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# For the /test endpoint below - hits Gemini 2.5 Flash directly, bypassing
# the graph, so it isolates the raw model from tool routing/prompting.
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
fast_llm = ChatGoogleGenerativeAI(model="gemma-4-26b-a4b-it", temperature=0)

# ============================================
# Lifespan: opens the checkpoint database connection ONCE, when the server
# actually starts, and closes it ONCE, when the server actually shuts down.
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # The chat-persistence database (RDS in deployment). Separate from the
    # backend SQL Server the store tools read - see checkpoint_db.py.
    print(f"[startup] chat persistence -> {describe_target()}")
    checkpointer_cm = AsyncPostgresSaver.from_conn_string(get_checkpoint_db_url())
    checkpointer = await checkpointer_cm.__aenter__()
    # Creates the checkpoint_* tables on first run if they don't exist yet,
    # so a fresh RDS instance needs no manual schema setup.
    await checkpointer.setup()

    app.state.checkpointer = checkpointer
    app.state.compiled_graph = graph.compile(checkpointer=checkpointer)

    # Warm the SQL sub-agent at startup. sql_agent_tool imports sql_ReAct
    # lazily (a module-level import there would be circular: sql_ReAct imports
    # tools), and building it - two chat models, bind_tools over 15 tools,
    # compiling the graph - costs ~2.9s. Without this, the first customer to
    # ask a data question pays that inside their request.
    import sql_ReAct  # noqa: F401

    # Open a couple of SQL Server connections now. The first connection to the
    # remote server costs ~0.9s (TCP + TLS + login) - better paid at boot than
    # inside the first customer's request. See db.py for the pool.
    from db import warm_pool
    warm_pool(2)

    yield

    await checkpointer_cm.__aexit__(None, None, None)


app = FastAPI(title="Grocery Ecommerce RAG Assistant", lifespan=lifespan)


@app.get("/")
def root():
    return {"status": "RAG assistant is running"}


@app.post("/chat", response_model=AnswerResponse)
async def chat(request: QuestionRequest, background_tasks: BackgroundTasks):
    thread_id = f"user-{request.user_id}"
    config = {"configurable": {"thread_id": thread_id, "user_id": request.user_id}}

    compiled_graph = app.state.compiled_graph
    result = await compiled_graph.ainvoke(
        {"messages": [{"role": "user", "content": request.question}]},
        config=config
    )
    # .text (not .content) - Gemini returns content as a list of blocks
    # (text + thought signature), not a plain string, which fails the
    # `answer: str` response schema.
    answer = result["messages"][-1].text

    background_tasks.add_task(summarize_chat, config, result)

    return {"question": request.question, "answer": answer}


@app.post("/terminate", response_model=TerminationResponse)
async def terminate_session(request: TerminationRequest):
    """
    Termination route - ends the session and deletes the LangGraph checkpoint
    thread for this user, returning the products they asked about during it.
    """
    thread_id = f"user-{request.user_id}"
    config = {"configurable": {"thread_id": thread_id, "user_id": request.user_id}}

    # Read the conversation BEFORE deleting it - adelete_thread wipes exactly
    # the checkpoints aget_state reads from.
    snapshot = await app.state.compiled_graph.aget_state(config)
    messages = (snapshot.values or {}).get("messages", [])
    questions = [
        m.content for m in messages
        if isinstance(m, HumanMessage) and isinstance(m.content, str)
    ]

    suggested_products = await get_suggested_product_ids(questions)

    await app.state.checkpointer.adelete_thread(thread_id)

    return {"user_id": request.user_id, "suggested_products": suggested_products}

@app.post("/test", response_model=AnswerResponse)
async def test(request: QuestionRequest):
    result = await llm.ainvoke(request.question)
    return {"question": request.question, "answer": result.text}