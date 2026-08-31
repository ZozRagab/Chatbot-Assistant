from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Depends, BackgroundTasks 
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from fastapi.responses import StreamingResponse
from schemas import QuestionRequest, AnswerResponse
from auth import create_token, get_current_user
from models import get_db, User
from utils import verify_password
from agent_graph import graph, DB_URI,summarize_chat


# ============================================
# Lifespan: opens the checkpoint database connection ONCE, when the server
# actually starts, and closes it ONCE, when the server actually shuts down.
# ============================================
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpointer_cm = AsyncPostgresSaver.from_conn_string(DB_URI)
    checkpointer = await checkpointer_cm.__aenter__()
    await checkpointer.setup()

    app.state.compiled_graph = graph.compile(checkpointer=checkpointer)

    yield

    await checkpointer_cm.__aexit__(None, None, None)


app = FastAPI(title="Grocery Ecommerce RAG Assistant", lifespan=lifespan)


@app.get("/")
def root():
    return {"status": "RAG assistant is running"}


@app.post("/login", status_code=status.HTTP_200_OK)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    # username field carries the email in this flow
    user = db.query(User).filter(User.Email == form.username).first()
    if not user or not verify_password(form.password, user.HashedPassword):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_token({"user_id": user.Id})
    return {"access_token": access_token, "token_type": "bearer"}
@app.post("/chat")
async def chat(request: QuestionRequest, current_user: User = Depends(get_current_user)):
    thread_id = f"user-{current_user.Id}"
    config = {"configurable": {"thread_id": thread_id, "user_id": current_user.Id}}
    compiled_graph = app.state.compiled_graph

    async def event_generator():
        async for chunk, metadata in compiled_graph.astream(
            {"messages": [{"role": "user", "content": request.question}]},
            config=config,
            stream_mode="messages"
        ):
            print(metadata, chunk.content)
            if getattr(chunk, "tool_call_chunks", None):
                continue
            if metadata.get("langgraph_node") != "ReAct_agent":   # only allow the OUTER agent's own node
                continue
            if chunk.content:
                yield chunk.content

    return StreamingResponse(event_generator(), media_type="text/plain")
@app.post("/terminate")
def terminate_session(current_user: User = Depends(get_current_user)):
    """
    Termination route
    """
    thread_id = f"user-{current_user.Id}"
    checkpointer.delete_thread(thread_id)
    return {"status": "terminated", "thread_id": thread_id}