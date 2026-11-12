
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Add the API directory to the Python import path.
API_DIR = Path(__file__).resolve().parents[1] / "protein-property-predictor"
sys.path.insert(0, str(API_DIR))

import api

client = TestClient(api.app)


def test_predict_valid_sequence(monkeypatch):
    sequence = "MALWMRLLPLLALLALWGPDPAAAFLVLGLVIGLIVG"

    mock_result = {
        "prediction": "membrane-bound",
        "membrane_probability": 0.85,
        "soluble_probability": 0.15,
        "threshold": 0.5,
        "sequence_length": len(sequence),
        "features": {"hydrophobic_fraction": 0.6},
        "model": "interpretable_sequence_baseline",
    }

    monkeypatch.setattr(api, "predict", lambda seq: mock_result)

    response = client.post(
        "/predict",
        json={"sequence": sequence},
    )

    assert response.status_code == 200

    data = response.json()

    expected_fields = {
        "prediction",
        "membrane_probability",
        "soluble_probability",
        "threshold",
        "sequence_length",
        "features",
        "model",
    }

    assert expected_fields.issubset(data.keys())
    assert data["prediction"] == "membrane-bound"
    assert data["sequence_length"] == len(sequence)


def test_predict_invalid_sequence():
    response = client.post(
        "/predict",
        json={"sequence": "MALWXB"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Unsupported amino-acid symbols: B, X"
    )
