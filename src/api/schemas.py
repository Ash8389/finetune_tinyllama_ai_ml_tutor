from pydantic import BaseModel

class LlmResponse(BaseModel):
    finetuned_answer: str
    basemodel_answer: str

class LlmRequest(BaseModel):
    question: str