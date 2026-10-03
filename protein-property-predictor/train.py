"""Train and evaluate an interpretable protein membrane-classification baseline.

This module is designed to run locally or on Domino. It avoids evaluating on the
training set, performs stratified holdout evaluation, cross-validation when the
sample size permits, logs artifacts to MLflow, and writes a self-describing model
bundle for inference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, clean_sequence, featurize_many

RANDOM_STATE = 42


def resolve_data_path(cli_path: str | None) -> Path:
    candidates = [
        cli_path,
        os.getenv("TRAIN_DATA"),
        "/mnt/netapp-volumes/snapshots/ppp-volume/2/train.csv",
        str(Path(__file__).parent / "data" / "train.csv"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "No training CSV found. Pass --data, set TRAIN_DATA, or place data/train.csv in the project."
    )


def resolve_output_dir(cli_output: str | None) -> Path:
    output = cli_output or os.getenv("MODEL_DIR") or "/mnt/artifacts/models/latest"
    path = Path(output)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        path = Path(__file__).parent / "data" / "models" / "latest"
        path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    required = {"sequence", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training data missing columns: {sorted(missing)}")

    df = df.dropna(subset=["sequence", "label"]).copy()
    df["sequence"] = df["sequence"].map(clean_sequence)
    df = df[df["sequence"].str.len() > 0].drop_duplicates(subset=["sequence"]).reset_index(drop=True)

    if df["label"].nunique() != 2:
        raise ValueError("Expected a binary label column with exactly two classes.")

    # Accept common textual labels while preserving 0/1 inputs.
    if not pd.api.types.is_numeric_dtype(df["label"]):
        normalized = df["label"].astype(str).str.lower().str.strip()
        mapping = {
            "soluble": 0,
            "membrane": 1,
            "membrane-bound": 1,
            "membrane_bound": 1,
            "0": 0,
            "1": 1,
        }
        if not normalized.isin(mapping).all():
            raise ValueError("Unsupported labels. Use soluble/membrane-bound or 0/1.")
        y = normalized.map(mapping).to_numpy(dtype=int)
    else:
        y = df["label"].to_numpy(dtype=int)

    X = featurize_many(df["sequence"].tolist())
    return df, X, y


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=2000,
                ),
            ),
        ]
    )


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    return float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None


def evaluate(model: Pipeline, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": safe_auc(y, prob),
        "average_precision": float(average_precision_score(y, prob)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
    }
    return metrics


def run_training(data_path: Path, output_dir: Path) -> Dict[str, Any]:
    df, X, y = load_dataset(data_path)
    class_counts = np.bincount(y, minlength=2)
    min_class = int(class_counts.min())

    if min_class < 2:
        raise ValueError("Need at least two examples per class for a stratified holdout split.")

    test_size = max(2, int(round(len(y) * 0.25)))
    test_size = min(test_size, len(y) - 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_pipeline()

    cv_summary: Dict[str, Any] = {"status": "skipped", "reason": "dataset too small"}
    train_min_class = int(np.bincount(y_train, minlength=2).min())
    folds = min(5, train_min_class)
    if folds >= 2:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        scoring = {
            "balanced_accuracy": "balanced_accuracy",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
        }
        result = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, error_score="raise")
        cv_summary = {
            "status": "completed",
            "folds": folds,
            **{
                metric.replace("test_", ""): {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
                for metric, values in result.items()
                if metric.startswith("test_")
            },
        }

    model.fit(X_train, y_train)
    holdout = evaluate(model, X_test, y_test)

    bundle = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "label_map": {0: "soluble", 1: "membrane-bound"},
        "threshold": 0.5,
        "training_rows": int(len(df)),
        "random_state": RANDOM_STATE,
    }
    model_path = output_dir / "model.joblib"
    joblib.dump(bundle, model_path)

    summary = {
        "task": "binary membrane-protein classification",
        "data_path": str(data_path),
        "n_samples": int(len(df)),
        "class_counts": {"soluble": int(class_counts[0]), "membrane_bound": int(class_counts[1])},
        "features": FEATURE_NAMES,
        "holdout": holdout,
        "cross_validation": cv_summary,
        "model_path": str(model_path),
    }
    summary_path = output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the protein property classifier")
    parser.add_argument("--data", help="CSV with sequence,label columns")
    parser.add_argument("--output", help="Directory for model.joblib and metrics.json")
    args = parser.parse_args()

    data_path = resolve_data_path(args.data)
    output_dir = resolve_output_dir(args.output)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Protein Property Predictor")

    with mlflow.start_run(run_name="interpretable_baseline") as run:
        summary = run_training(data_path, output_dir)
        mlflow.log_param("model", "StandardScaler+LogisticRegression")
        mlflow.log_param("feature_count", len(FEATURE_NAMES))
        mlflow.log_param("training_rows", summary["n_samples"])
        for key in ["accuracy", "balanced_accuracy", "f1", "roc_auc", "average_precision"]:
            value = summary["holdout"].get(key)
            if value is not None:
                mlflow.log_metric(f"holdout_{key}", value)
        mlflow.log_artifact(str(output_dir / "model.joblib"), artifact_path="model")
        mlflow.log_artifact(str(output_dir / "metrics.json"), artifact_path="evaluation")
        print(json.dumps(summary, indent=2))
        print(f"MLflow run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
