from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model import predict

st.set_page_config(
    page_title="Protein Property Predictor",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Protein Property Predictor")
st.caption(
    "An interpretable sequence-based classifier for soluble vs. membrane-associated proteins. "
    "Built as a deployment-ready ML reference workflow with validation, MLflow tracking, API serving, and tests."
)

EXAMPLES = {
    "Hydrophobic membrane-like": "MALWMRLLPLLALLALWGPDPAAAFLVLGLVIGLIVG",
    "Soluble enzyme-like": "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRG",
    "Custom": "",
}

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Sequence")
    preset = st.selectbox("Example", list(EXAMPLES), index=0)
    default = EXAMPLES[preset]
    seq = st.text_area(
        "Paste an amino-acid sequence or FASTA record",
        value=default,
        height=220,
        placeholder=">protein_1\nMALWMRLLPLLALLALWGPDPAAA...",
    )
    submitted = st.button("Run prediction", type="primary", use_container_width=True)

with right:
    st.subheader("What the model uses")
    st.markdown(
        """
        The baseline is deliberately **interpretable** rather than a black box. It uses:

        - sequence length
        - global and N-terminal hydrophobicity
        - maximum 21-aa hydrophobic window
        - charged / aromatic / small-residue fractions
        - a simple net-charge proxy
        - helix-breaker fraction
        - sequence entropy

        The repository separates **feature engineering, training, evaluation, inference, UI, and API serving** so each layer can be tested independently.
        """
    )

if submitted:
    try:
        result = predict(seq)
        st.success(f"Prediction: **{result['prediction']}**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Membrane probability", f"{result['membrane_probability']:.1%}")
        m2.metric("Soluble probability", f"{result['soluble_probability']:.1%}")
        m3.metric("Sequence length", result["sequence_length"])

        st.subheader("Interpretable sequence features")
        feature_df = pd.DataFrame(
            [{"feature": key, "value": value} for key, value in result["features"].items()]
        )
        st.dataframe(feature_df, hide_index=True, use_container_width=True)

        st.info(
            "This is a portfolio / engineering demonstration, not a substitute for experimentally validated "
            "subcellular-localization tools or biological annotation."
        )
    except Exception as exc:
        st.error(str(exc))

with st.expander("Architecture"):
    st.code(
        "Sequence / FASTA\n"
        "      ↓\n"
        "validation + feature engineering\n"
        "      ↓\n"
        "StandardScaler → LogisticRegression\n"
        "      ↓\n"
        "holdout + stratified CV + MLflow\n"
        "      ↓\n"
        "model bundle → Streamlit / FastAPI / CLI",
        language="text",
    )

st.caption("Synthetic/demo-scale data may be used in the repository. No patient, proprietary, or confidential data is included.")
