from fastapi import APIRouter, Depends
from src.api.services import Service
from src.api.dependencies import get_service

router = APIRouter(
    prefix="/chat",
    tags=["AI/ML finetuned model api"]
)

@router.get("")
def chat(q:str, service: Service = Depends(get_service)):
    return service.query_service(question=q)