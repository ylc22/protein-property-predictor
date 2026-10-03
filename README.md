# 🧬 Protein Property Predictor

[![CI](https://github.com/ylc22/protein-property-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/ylc22/protein-property-predictor/actions/workflows/ci.yml)

An end-to-end scientific ML project for classifying protein sequences as **soluble** or **membrane-associated** — built to demonstrate not just modeling, but the complete path from sequence validation and feature engineering to evaluation, experiment tracking, API serving, containerization, and interactive deployment.

> **Why this project exists:** many portfolio ML repos stop at `model.fit()`. This one is structured to show how I think about a scientific model as a product: reproducible inputs, interpretable features, honest evaluation, versioned artifacts, multiple serving layers, testing, and deployment.

## What this repository demonstrates

- **Scientific feature engineering** from raw amino-acid sequence
- **Leakage-aware workflow structure** with a separate holdout set and stratified cross-validation
- **Interpretable baseline modeling** with scaling + class-balanced logistic regression
- **Evaluation beyond accuracy**: balanced accuracy, F1, ROC AUC, average precision, confusion matrix, per-class metrics
- **MLflow experiment tracking** for metrics and model artifacts
- **Self-describing model bundles** for reproducible inference
- **Streamlit application** for interactive prediction and feature inspection
- **FastAPI service** with typed request/response contracts
- **Dockerized serving**
- **Automated tests + GitHub Actions CI**
- **Model card** covering intended use, limitations, and responsible interpretation

---

## Architecture

```text
amino-acid sequence / FASTA
          │
          ▼
  validation + cleaning
          │
          ▼
 interpretable sequence features
          │
          ├── global / N-terminal hydrophobicity
          ├── max 21-aa hydrophobic window
          ├── charge / aromatic / small-residue fractions
          ├── helix-breaker fraction
          └── sequence entropy
          │
          ▼
 StandardScaler → LogisticRegression
          │
          ├── stratified holdout evaluation
          ├── stratified cross-validation
          └── MLflow tracking
          │
          ▼
     model bundle
       /      \
      ▼        ▼
 Streamlit   FastAPI
    UI       service
              │
              ▼
            Docker
```

---

## Why an interpretable baseline?

Membrane-associated proteins often contain strong hydrophobic segments, particularly transmembrane helices or signal-like regions. Rather than hiding that biological intuition inside a black box, this baseline makes it explicit and testable.

The current model uses ten features:

| Feature | Motivation |
|---|---|
| log sequence length | stabilizes scale while preserving size information |
| global hydrophobic fraction | overall membrane affinity signal |
| N-terminal hydrophobic fraction | captures signal-peptide / anchor-like behavior |
| maximum 21-aa hydrophobic window | approximates a transmembrane-helix-sized local segment |
| charged fraction | contrasts hydrophobic membrane cores with polar/charged composition |
| net-charge proxy | coarse electrostatic signal |
| aromatic fraction | membrane interfaces are often enriched in aromatic residues |
| small-residue fraction | local packing / composition signal |
| helix-breaker fraction | proline/glycine can disrupt long hydrophobic helices |
| sequence entropy | coarse sequence-complexity measure |

This is **not presented as state-of-the-art biology**. It is an interpretable systems baseline designed to support rigorous benchmarking and deployment. The next modeling layer can compare it against protein-language-model embeddings under the same evaluation harness.

---

## Repository structure

```text
.
├── .github/workflows/ci.yml        # automated tests / import checks
├── Dockerfile                      # containerized FastAPI service
├── MODEL_CARD.md                   # intended use, limitations, evaluation notes
├── app.sh                          # Domino app entrypoint
├── tests/
│   └── test_features.py            # feature parsing and bounds tests
└── protein-property-predictor/
    ├── features.py                 # validated sequence feature engineering
    ├── train.py                    # holdout + CV + MLflow training pipeline
    ├── model.py                    # reusable inference API / CLI
    ├── app.py                      # portfolio-quality Streamlit application
    ├── api.py                      # FastAPI service
    ├── data/                       # demo data / local artifacts
    └── env/requirements.txt
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/ylc22/protein-property-predictor.git
cd protein-property-predictor
python -m venv .venv
source .venv/bin/activate
pip install -r protein-property-predictor/env/requirements.txt
```

### 2. Train

The trainer accepts any CSV containing:

```text
sequence,label
MALWMRLLPLLALLALWGPDPAAA,membrane-bound
MSTNPKPQRKTKRNTNRRPQDVK,soluble
```

Run:

```bash
python protein-property-predictor/train.py \
  --data protein-property-predictor/data/train.csv \
  --output protein-property-predictor/data/models/latest
```

The run writes:

- `model.joblib` — model + feature metadata + threshold
- `metrics.json` — holdout metrics, class counts, feature list, CV summary
- MLflow artifacts / metrics when a tracking server is available

### 3. Predict from CLI

```bash
MODEL_PATH=protein-property-predictor/data/models/latest/model.joblib \
python protein-property-predictor/model.py \
"MALWMRLLPLLALLALWGPDPAAAFLVLGLVIGLIVG"
```

### 4. Launch the Streamlit app

```bash
MODEL_PATH=protein-property-predictor/data/models/latest/model.joblib \
streamlit run protein-property-predictor/app.py
```

### 5. Run the API

```bash
cd protein-property-predictor
MODEL_PATH=data/models/latest/model.joblib \
uvicorn api:app --reload
```

Then:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence":"MALWMRLLPLLALLALWGPDPAAAFLVLGLVIGLIVG"}'
```

Interactive API docs are available at `/docs` while the service is running.

### 6. Run tests

```bash
pytest -q
```

---

## Training and evaluation design

The original demo version of this project trained and evaluated on the same tiny CSV. That is useful for showing infrastructure, but not a defensible modeling workflow.

The current pipeline instead:

1. validates and de-duplicates sequences
2. checks that both classes are represented
3. performs a stratified train / holdout split
4. runs stratified cross-validation when class counts permit
5. fits only on the training partition
6. reports holdout balanced accuracy, F1, ROC AUC, average precision, confusion matrix, and per-class metrics
7. serializes the exact preprocessing + classifier pipeline together

For a serious biological benchmark, the next step is **homology-aware splitting** rather than random sequence splitting, because near-duplicate families across train/test can inflate performance.

---

## Deployment paths

### Local / Docker

```bash
docker build -t protein-property-predictor .
docker run -p 8000:8000 \
  -e MODEL_PATH=/app/protein-property-predictor/data/models/latest/model.joblib \
  protein-property-predictor
```

### Domino

The repo preserves the original Domino workflow concept:

```text
Workspace → mounted data → training job → MLflow → model artifact → App / Endpoint
```

Paths are now configurable rather than hard-coded wherever practical:

- `TRAIN_DATA` — optional training CSV path
- `MODEL_DIR` — optional output directory
- `MODEL_PATH` — inference model bundle
- `MLFLOW_TRACKING_URI` — MLflow backend

---

## Responsible interpretation

This project is an ML engineering and computational biology portfolio demonstration. It is **not** intended for clinical use, therapeutic decisions, experimental annotation, or claims of state-of-the-art membrane-protein prediction.

See [`MODEL_CARD.md`](MODEL_CARD.md) for limitations, evaluation caveats, and the recommended next modeling step.

---

## Roadmap

- [x] reusable feature engineering module
- [x] stratified holdout evaluation
- [x] cross-validation
- [x] MLflow experiment tracking
- [x] Streamlit interface
- [x] FastAPI endpoint
- [x] Docker container
- [x] tests + CI
- [x] model card
- [ ] homology-aware / cluster-aware train-test splitting
- [ ] larger public benchmark dataset
- [ ] frozen protein-language-model embeddings (ESM-family or equivalent)
- [ ] baseline-vs-embedding benchmark report
- [ ] calibration and uncertainty diagnostics on a sufficiently large validation cohort

---

## Background

This project originated as a life-sciences solutions-engineering demo focused on the **ML platform workflow** around a simple protein model. I later expanded it into a more complete public reference implementation to better represent how I approach scientific ML systems: simple baselines first, honest evaluation, clear interfaces, reproducibility, and a path toward stronger models.

**Author:** Luis Chan  
Computational Biology · Machine Learning · Bioinformatics
