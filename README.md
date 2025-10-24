# 🧬 Protein Property Predictor (Toy Model) — Domino Demo by Luis Chan

Classifies protein sequences as **soluble** or **membrane-bound**.  
Designed to showcase the Domino flow using a **NetApp Volume Snapshot** for data:  
**Workspace → NetApp Volume Snapshot → Training Job → App (and later Endpoint).**
---

## 🚀 What’s inside

- **Two predictors**
  - **Rule-based** (no training): thresholds on hydrophobicity.
  - **ML** (logistic regression): trained on a tiny toy CSV.
- **NetApp Volume–based I/O**  
  Reads training data directly from `/mnt/netapp-volumes/snapshots/ppp-volume/2/train.csv`  
  and saves model artifacts to `/mnt/artifacts`.
- **Simple UI**: Streamlit app publishable as a Domino App.

---

## 🗂️ Repo structure

property-property-model/
├─ app.sh # Domino App entry (MUST be at repo root)
├─ README.md
└─ protein-property-predictor/
├─ app.py # Streamlit UI
├─ model.py # Prediction (rule + ML loader)
├─ train.py # Training (logistic regression)
├─ data/
│ └─ train.csv # just for backup
├─ env/
│ └─ requirements.txt # numpy, pandas, scikit-learn, joblib, streamlit, requests
└─ models/
└─ latest/ # Trained model is written here (or in Dataset)

📁 **Training data** is located in the mounted NetApp volume:  
`/mnt/netapp-volumes/snapshots/ppp-volume/2/train.csv`

---

## 🧪 Biology (60-second version)

- Proteins are chains of 20 amino acids (letters **A, C, D, …, Y**).
- Water is **hydrophilic**; cell membranes are **hydrophobic** (oily).
- Hydrophobic letters **A I L M F W V Y** tend to favor membranes.
- Many membrane/secreted proteins begin with a **hydrophobic N-terminus** (~first 20 aa).
- We compute features:
  1) **Overall hydrophobic fraction**, 2) **N-terminal hydrophobic fraction**, 3) **Length**  
  → Predict **membrane-bound** vs **soluble**.

**Why care?** Membrane proteins (receptors, channels, pumps) are major drug targets and behave differently in experiments than soluble proteins.

---

## 📦 Requirements

- Domino project connected to this GitHub repo.
- A Domino **Compute Environment** with Python 3 & `pip`.
- **NetApp Volume Snapshot** mounted at: `/mnt/netapp-volumes/snapshots/ppp-volume/2/train.csv`  
- Python packages (already in `env/requirements.txt`):



---

## 🧑‍💻 Quickstart on Domino

### 1) Connect code & mount a Dataset
1. **Project → Code** → add this Git repo (prefer “Use latest from external repo”).
2. **NetApp Volume Snapshot** mounted at: `/mnt/netapp-volumes/snapshots/ppp-volume/2/train.csv`  

### 2) Workspace: set env & install deps
Open a terminal **at the repo root**:
```bash

# Install packages to user site
pip3 install --user -r protein-property-predictor/env/requirements.txt
export PATH="$HOME/.local/bin:$PATH"


# Protein Property Predictor — Quick Ops Runbook

## 3) **NetApp Volume Snapshot** mounted at: `/mnt/netapp-volumes/snapshots/ppp-volume/2/train.csv`  

## 4) Train the model (writes `model.joblib`)
```bash
cd protein-property-predictor
python3 train.py
# → prints JSON with model_path, samples, etc.
```

## 5) Predict from CLI
**Force ML** (uses model saved in `/mnt/netapp-volumes/snapshots/ppp-volume/2/train.csv`)
```bash
python3 model.py --seq "MKKLLLLLLLLLALALALAAAGAGA" --mode ml
```

**Force rule** (no training needed)
```bash
python3 model.py --seq "MSTNPKPQRKTKRNTNRRPQDVK" --mode rule
```

**Auto: try ML, fallback to rule**
```bash
python3 model.py --seq ">p\nMAALALLLGVVVVALAAA" --mode auto
```

---

## 🖥️ Publish the Streamlit App (Domino Apps)

### Files used
- `app.sh` (repo root, Domino App entry script)
- `protein-property-predictor/app.py` (Streamlit UI)

### Steps
1. **Deploy → Apps → New App**
2. **Name:** `ppp-app`
3. **Entry script:** `app.sh` *(must be at repo root)*
4. **Compute environment:** any Python env with pip
5. **Datasets:** mount the same Dataset snapshot used for training
6. **Environment variables:**
   ```bash
   DATASET_DIR=/domino/datasets/local/protein-property-predictor   # adjust to your path
   ```
7. **Publish / Launch** → open the App URL.

---

## 🧠 How it works

### `train.py`
- Loads CSV from **NetApp Volume Snapshot** mounted at: `/mnt/netapp-volumes/snapshots/ppp-volume/2/train.csv`  
- Featurizes sequences (overall & N-terminal hydrophobicity, length).
- Trains a `LogisticRegression`; saves to `${DATASET_DIR or ./}/models/latest/model.joblib`.

### `model.py`
- **Rule mode:** simple thresholds on hydrophobicity features.
- **ML mode:** loads `model.joblib` and returns probability + label.
- **Auto mode:** try ML; if no model available, fallback to rule.

### `app.py` (Streamlit)
- Text area for sequence / tiny FASTA.
- Mode selector (`auto` | `ml` | `rule`).
- Calls `predict()` and renders JSON.

---

## 🧭 Troubleshooting

### App error: “entry script './app.sh' not found”
- Ensure `app.sh` exists at **repo root** (top level in Project → Code).
- In **Apps → Edit**, set **Entry script** exactly to `app.sh`.

### Blank Streamlit page
- Check **App logs** in Domino.
- Ensure `streamlit` is installed & on PATH (handled in `app.sh`).
- Workspace health check:
  ```bash
  export DOMINO_APP_PORT=8501
  streamlit run protein-property-predictor/app.py --server.port $DOMINO_APP_PORT --server.address 0.0.0.0 &
  sleep 3
  curl -s http://127.0.0.1:$DOMINO_APP_PORT/_stcore/health
  ```

### ML mode: “model not found”
- Run `python3 train.py` first.
- Confirm printed `model_path` exists under `${DATASET_DIR}/models/latest/` (or `./models/latest/`).

### Git push from Workspace fails
- Configure identity:
  ```bash
  git config --global user.name "YOUR_GH_USERNAME"
  git config --global user.email "you@company.com"
  ```
- Use a GitHub PAT; complete SAML SSO if prompted for org repos.

---

## 🗺️ Roadmap (next steps)
- Domino Endpoint that wraps `predict()` with request/response logging.(completed✅)
- Jobs to schedule `train.py` and snapshot the Dataset.(completed✅)
- MLflow logging of params/metrics/artifacts.(completed✅)
- Tests for parsing/featurization & golden predictions.
- Expand dataset; add metrics (confusion matrix, ROC/PR).

