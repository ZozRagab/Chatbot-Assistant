from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    route: str
    answer: str


class TokenData(BaseModel):
    user_id: int