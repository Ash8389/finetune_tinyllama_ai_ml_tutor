from fastapi import Request

from src.api.services import Service


def get_service(request: Request):
    return Service(
        base_model = request.app.state.base_model,
        finetuned_model = request.app.state.finetuned_model,
        tokenizer = request.app.state.tokenizer
    )