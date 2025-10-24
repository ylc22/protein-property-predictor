import os, sys, streamlit as st

# Ensure we can import model.py in the same folder even though the folder name has a hyphen
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model import predict  # uses your existing predict(input_text, mode='auto')

st.set_page_config(page_title="Protein Property Predictor", layout="centered")
st.title("Protein Property Predictor (toy)")

seq = st.text_area("Paste a sequence or tiny FASTA:", height=140, value="MKKLLLLLLLLLALALALAAAGAGA")
mode = st.selectbox("Mode", ["auto", "ml", "rule"], index=0)

if st.button("Predict"):
    out = predict(seq, mode=mode)
    st.json(out)

st.caption("Backed by Domino Dataset via DATASET_DIR; model trained by train.py (logistic regression).")
