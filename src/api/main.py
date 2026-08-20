from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.api.routes import router

from src.inference.pipeline import pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading models...")

    (
        app.state.base_model,
        app.state.finetuned_model,
        app.state.tokenizer
    ) = pipeline()

    print("Models loaded.")

    yield

    print("Shutting down...")

app = FastAPI(
    title="Ai/Ml tutor finetuned model api",
    description="Ai/Ml tutor finetuned model api",
    version="1.0",
    lifespan=lifespan
)

app.include_router(router)