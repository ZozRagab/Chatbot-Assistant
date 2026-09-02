from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str
    user_id: str


class AnswerResponse(BaseModel):
    question: str
    answer: str


class TerminationRequest(BaseModel):
    user_id: str


class TerminationResponse(BaseModel):
    user_id: str
    suggested_products: list[int] = [] 