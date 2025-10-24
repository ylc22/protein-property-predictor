# model.py — inference only (used by app.py)
#
# Loads model.joblib (trained via train.py) and provides predict() for Streamlit app.

import os
import joblib
import numpy as np

# ---------- Feature functions ----------
HYDRO = set("AILMFWVY")

def _clean(s):
    return (s or "").upper().replace(" ", "").replace("\n", "").replace("\r", "")

def _hyd_frac(s):
    s = _clean(s)
    return sum(1 for c in s if c in HYDRO) / max(len(s), 1)

def _nterm_hyd_frac(s, w=20):
    s = _clean(s)
    return _hyd_frac(s[:w])

def featurize(seq):
    seq = _clean(seq)
    hyd = _hyd_frac(seq)
    nterm_hyd = _nterm_hyd_frac(seq)
    length = float(len(seq))
    X = np.array([[hyd, nterm_hyd, length]])
    return X, {"length": int(length), "hydrophobic_fraction": round(hyd, 3), "nterm_hydrophobic_fraction": round(nterm_hyd, 3)}


# ---------- Resolve model path ----------
DATASET_DIR = os.getenv("DATASET_DIR", "").strip()
HERE = os.path.dirname(__file__)


MODEL_PATH = "/mnt/artifacts/models/latest/model.joblib"


# ---------- Load trained model ----------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Model file not found at {MODEL_PATH}. Run train.py first."
    )

MODEL = joblib.load(MODEL_PATH)

# ---------- Prediction ----------
def predict(seq: str, mode: str = "auto") -> dict:
    """
    Predicts protein property from amino acid sequence.

    Returns:
        dict with prediction label, confidence, and features.
    """
    seq = _clean(seq)
    if not seq:
        return {"error": "Empty sequence."}

    # Compute features
    X, feats = featurize(seq)

    if mode == "rule":
        hyd = feats["hydrophobic_fraction"]
        pred_label = "membrane-bound" if hyd > 0.45 else "soluble"
        return {
            "prediction": pred_label,
            "confidence": round(hyd, 3),
            "features": feats,
            "mode": "rule-based"
        }

    # ML-based prediction
    try:
        prob = float(MODEL.predict_proba(X)[0, 1])
        pred_label = "membrane-bound" if prob >= 0.5 else "soluble"
    except Exception as e:
        return {"error": f"Model prediction failed: {str(e)}"}

    return {
        "prediction": pred_label,
        "confidence": round(prob, 3),
        "features": feats,
        "mode": "ml" if mode == "ml" else "auto"
    }
