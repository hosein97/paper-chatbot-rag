from pydantic import BaseModel

class ChatRequest(BaseModel):
    filename: str
    question: str

class ChatResponse(BaseModel):
    answer: str
