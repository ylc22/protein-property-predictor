"""Feature engineering for protein sequence classification.

The features are intentionally interpretable and dependency-light so the demo can
run in constrained notebook, Domino, API, and CI environments.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWVY")
AROMATIC = set("FWY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
SMALL = set("AGST")
HELIX_BREAKERS = set("PG")

FEATURE_NAMES = [
    "length_log",
    "hydrophobic_fraction",
    "nterm_hydrophobic_fraction",
    "max_window_hydrophobic_fraction",
    "charged_fraction",
    "net_charge_proxy",
    "aromatic_fraction",
    "small_residue_fraction",
    "helix_breaker_fraction",
    "sequence_entropy",
]


def clean_sequence(raw: str) -> str:
    """Normalize plain sequence or FASTA input and validate amino-acid symbols."""
    if raw is None:
        return ""
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    lines = [line for line in lines if not line.startswith(">")]
    seq = "".join(lines).replace(" ", "").upper()
    invalid = sorted(set(seq) - AMINO_ACIDS)
    if invalid:
        raise ValueError(f"Unsupported amino-acid symbols: {', '.join(invalid)}")
    return seq


def _fraction(seq: str, residue_set: set[str]) -> float:
    return sum(aa in residue_set for aa in seq) / max(len(seq), 1)


def _window_max_fraction(seq: str, residue_set: set[str], window: int = 21) -> float:
    if not seq:
        return 0.0
    if len(seq) <= window:
        return _fraction(seq, residue_set)
    hits = [1 if aa in residue_set else 0 for aa in seq]
    running = sum(hits[:window])
    best = running
    for idx in range(window, len(hits)):
        running += hits[idx] - hits[idx - window]
        best = max(best, running)
    return best / window


def _entropy(seq: str) -> float:
    if not seq:
        return 0.0
    counts = Counter(seq)
    n = len(seq)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def featurize_sequence(raw: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return model-ready vector plus human-readable feature dictionary."""
    seq = clean_sequence(raw)
    if not seq:
        raise ValueError("Sequence is empty after parsing.")

    nterm = seq[:25]
    positive = _fraction(seq, POSITIVE)
    negative = _fraction(seq, NEGATIVE)

    features = {
        "length_log": math.log1p(len(seq)),
        "hydrophobic_fraction": _fraction(seq, HYDROPHOBIC),
        "nterm_hydrophobic_fraction": _fraction(nterm, HYDROPHOBIC),
        "max_window_hydrophobic_fraction": _window_max_fraction(seq, HYDROPHOBIC, window=21),
        "charged_fraction": _fraction(seq, POSITIVE | NEGATIVE),
        "net_charge_proxy": positive - negative,
        "aromatic_fraction": _fraction(seq, AROMATIC),
        "small_residue_fraction": _fraction(seq, SMALL),
        "helix_breaker_fraction": _fraction(seq, HELIX_BREAKERS),
        "sequence_entropy": _entropy(seq),
    }
    vector = np.array([features[name] for name in FEATURE_NAMES], dtype=float)
    return vector, features


def featurize_many(sequences: List[str]) -> np.ndarray:
    return np.vstack([featurize_sequence(seq)[0] for seq in sequences])
