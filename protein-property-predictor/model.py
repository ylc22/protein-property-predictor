"""Inference API for the Protein Property Predictor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import joblib

from features import clean_sequence, featurize_sequence

HERE = Path(__file__).resolve().parent


def resolve_model_path() -> Path:
    candidates = [
        os.getenv("MODEL_PATH"),
        "/mnt/artifacts/models/latest/model.joblib",
        str(HERE / "data" / "models" / "latest" / "model.joblib"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "Model bundle not found. Train with `python protein-property-predictor/train.py --data ...` "
        "or set MODEL_PATH."
    )


def load_bundle() -> Dict[str, Any]:
    bundle = joblib.load(resolve_model_path())
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle
    # Backward compatibility with the original demo's bare estimator.
    return {
        "model": bundle,
        "label_map": {0: "soluble", 1: "membrane-bound"},
        "threshold": 0.5,
        "feature_names": ["legacy_features"],
    }


def predict(sequence: str) -> Dict[str, Any]:
    seq = clean_sequence(sequence)
    if not seq:
        raise ValueError("Provide a non-empty amino-acid sequence or FASTA record.")

    vector, features = featurize_sequence(seq)
    bundle = load_bundle()
    model = bundle["model"]
    threshold = float(bundle.get("threshold", 0.5))
    probability = float(model.predict_proba(vector.reshape(1, -1))[0, 1])
    encoded = int(probability >= threshold)
    label_map = bundle.get("label_map", {0: "soluble", 1: "membrane-bound"})

    # joblib can preserve integer or string dict keys depending on serialization path.
    label = label_map.get(encoded, label_map.get(str(encoded), str(encoded)))

    return {
        "prediction": label,
        "membrane_probability": round(probability, 4),
        "soluble_probability": round(1.0 - probability, 4),
        "threshold": threshold,
        "sequence_length": len(seq),
        "features": {k: round(float(v), 4) for k, v in features.items()},
        "model": "interpretable_sequence_baseline",
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")
    args = parser.parse_args()
    print(json.dumps(predict(args.sequence), indent=2))
