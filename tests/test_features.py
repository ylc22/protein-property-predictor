import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "protein-property-predictor"
sys.path.insert(0, str(ROOT))

from features import clean_sequence, featurize_sequence


def test_clean_sequence_accepts_fasta():
    assert clean_sequence(">x\nMALW\nMRLL") == "MALWMRLL"


def test_clean_sequence_rejects_invalid_symbols():
    try:
        clean_sequence("MALWXB")
    except ValueError as exc:
        assert "Unsupported amino-acid symbols" in str(exc)
    else:
        raise AssertionError("Expected invalid amino acid symbols to raise")


def test_features_are_bounded_where_expected():
    _, features = featurize_sequence("MALWMRLLPLLALLALWGPDPAAA")
    for name in [
        "hydrophobic_fraction",
        "nterm_hydrophobic_fraction",
        "max_window_hydrophobic_fraction",
        "charged_fraction",
        "aromatic_fraction",
        "small_residue_fraction",
        "helix_breaker_fraction",
    ]:
        assert 0.0 <= features[name] <= 1.0
