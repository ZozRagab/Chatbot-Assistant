from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from schemas import QuestionRequest, AnswerResponse, TerminationRequest, TerminationResponse
from agent_graph import graph, DB_URI, summarize_chat
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


# ============================================
# Lifespan: opens the checkpoint database connection ONCE, when the server
# actually starts, and closes it ONCE, when the server actually shuts down.
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpointer_cm = AsyncPostgresSaver.from_conn_string(DB_URI)
    checkpointer = await checkpointer_cm.__aenter__()
    await checkpointer.setup()

    app.state.checkpointer = checkpointer
    app.state.compiled_graph = graph.compile(checkpointer=checkpointer)

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
    answer = result["messages"][-1].content

    background_tasks.add_task(summarize_chat, config, result)

    return {"question": request.question, "answer": answer}


@app.post("/terminate", response_model=TerminationResponse)
async def terminate_session(request: TerminationRequest):
    """
    Termination route - deletes the LangGraph checkpoint thread for this user.
    """
    thread_id = f"user-{request.user_id}"
    checkpointer = app.state.checkpointer
    await checkpointer.adelete_thread(thread_id)
    return {"status": "terminated", "thread_id": thread_id}