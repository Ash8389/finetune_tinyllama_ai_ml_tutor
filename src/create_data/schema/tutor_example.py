from pydantic import BaseModel
from typing import Literal

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class TutorExample(BaseModel):
    id: str
    topic: str
    domain: str
    category: str
    difficulty: Literal[
        "beginner",
        "intermediate",
        "advanced"
    ]
    messages: list[Message]