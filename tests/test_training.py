import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "protein-property-predictor"
sys.path.insert(0, str(ROOT))

from train import run_training


def test_training_pipeline_writes_bundle_and_metrics(tmp_path):
    membrane = [
        "MALWMRLLPLLALLALWGPDPA",
        "MFLVLLPLVSSQCVNLTTRT",
        "MKWVTFLLLLFSSAYSRGVFR",
        "MLGLLLLPLLWAGALAMEPA",
        "MAVLLLLLLLAGALAAGAEA",
        "MFFLLLLVLATATGVHSADA",
        "MALALLLLVAGVANAEEAEA",
        "MPLLLLLAAGVAAAPAAAAA",
    ]
    soluble = [
        "MSTNPKPQRKTKRNTNRRPQDVK",
        "MADQLTEEQIAEFKEAFSLFDKD",
        "MSDSEVNQEAKPEVKPEVKPAA",
        "MARGKKIGYSAPRQTKEAATKAA",
        "MSEYIRKSLDQLNEKVRQLEEQA",
        "MNAEKRRLVQKAKLAEQAERYD",
        "MSDLKDKAKKLEAAGVEVEVKP",
        "MARGKKQVEQLKQQLEQLEKQR",
    ]
    df = pd.DataFrame(
        {
            "sequence": membrane + soluble,
            "label": ["membrane-bound"] * len(membrane) + ["soluble"] * len(soluble),
        }
    )
    data_path = tmp_path / "train.csv"
    output_dir = tmp_path / "model"
    df.to_csv(data_path, index=False)

    summary = run_training(data_path, output_dir)

    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()
    assert summary["n_samples"] == 16
    assert "balanced_accuracy" in summary["holdout"]

    persisted = json.loads((output_dir / "metrics.json").read_text())
    assert persisted["task"] == "binary membrane-protein classification"
