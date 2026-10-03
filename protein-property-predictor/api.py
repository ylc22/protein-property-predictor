"""FastAPI service for production-style inference demos."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import predict

app = FastAPI(
    title="Protein Property Predictor API",
    version="2.0.0",
    description="Interpretable soluble vs. membrane protein sequence classifier.",
)


class PredictionRequest(BaseModel):
    sequence: str = Field(..., min_length=1, description="Amino-acid sequence or FASTA record")


class PredictionResponse(BaseModel):
    prediction: str
    membrane_probability: float
    soluble_probability: float
    threshold: float
    sequence_length: int
    features: dict[str, float]
    model: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(payload: PredictionRequest) -> PredictionResponse:
    try:
        return PredictionResponse(**predict(payload.sequence))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
