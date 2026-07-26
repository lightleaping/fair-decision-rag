"""Official Track 2 HTTP API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.submission_service import SubmissionService


logger = logging.getLogger(__name__)
service: SubmissionService | None = None


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    question: str


class PredictResponse(BaseModel):
    id: str
    retrieved_chunk_ids: list[str]
    answer: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    service = SubmissionService.from_environment()
    logger.info("Submission service ready in %.3fs", service.startup_seconds)
    yield
    service = None


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict[str, str]:
    if service is None:
        raise HTTPException(status_code=503, detail="initializing")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="initializing")
    try:
        result = service.predict(request.id, request.question)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return PredictResponse(**result)
