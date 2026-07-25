from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from schemas import QuestionRequest, AnswerResponse
from main import answer
from auth import create_token, get_current_user
from models import get_db, Customer
from utils import verify_password

# ============================================
# All the heavy setup (embedding model, ChromaDB connection, LLM clients)
# happens ONCE here, at import time - when the server starts.
# ============================================

app = FastAPI(title="Ecommerce RAG Assistant")


@app.get("/")
def root():
    return {"status": "RAG assistant is running"}


@app.post("/login", status_code=status.HTTP_200_OK)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    # username field carries the email in this flow
    customer = db.query(Customer).filter(Customer.email == form.username).first()
    if not customer or not verify_password(form.password, customer.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_token({"user_id": customer.id})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest, current_customer: Customer = Depends(get_current_user)):
    question = request.question
    route, response_text = answer(question, customer_id=current_customer.id)

    return AnswerResponse(
        question=question,
        route=route,
        answer=response_text
    )