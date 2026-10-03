# Model Card — Protein Property Predictor

## Summary

This repository demonstrates an end-to-end machine-learning workflow for binary classification of protein sequences into **soluble** vs. **membrane-associated** classes. The current default model is an interpretable baseline using engineered sequence features and logistic regression.

The project is designed as a portfolio-quality ML systems example: data validation, feature engineering, reproducible training, holdout evaluation, stratified cross-validation, MLflow tracking, model serialization, Streamlit UI, FastAPI serving, containerization, and CI.

## Intended use

- Demonstrate scientific ML workflow design and deployment patterns.
- Provide an interpretable baseline before introducing heavier sequence encoders.
- Support local, Domino, and containerized inference demos.

## Not intended for

- Clinical, diagnostic, therapeutic, regulatory, or laboratory decision-making.
- Production biological annotation without validation on an appropriate external benchmark.
- Claims of state-of-the-art membrane-protein prediction.

## Inputs and outputs

**Input:** amino-acid sequence or a single FASTA record using the 20 canonical residues.

**Output:** predicted class, membrane probability, soluble probability, threshold, sequence length, and interpretable sequence features.

## Features

The baseline uses ten interpretable features:

1. log sequence length
2. global hydrophobic fraction
3. N-terminal hydrophobic fraction
4. maximum 21-residue hydrophobic-window fraction
5. charged-residue fraction
6. net-charge proxy
7. aromatic-residue fraction
8. small-residue fraction
9. helix-breaker fraction
10. sequence entropy

## Evaluation design

Training uses a stratified holdout split and, when sample counts permit, stratified cross-validation. Reported artifacts can include balanced accuracy, F1, ROC AUC, average precision, confusion matrix, and per-class metrics.

Metrics are only meaningful for the dataset used in a particular run. This repository intentionally does not present toy-data performance as biological evidence.

## Limitations

- Hand-engineered features cannot capture the full sequence, structural, evolutionary, or contextual information available to modern protein language models.
- Performance is sensitive to dataset construction, redundancy, class balance, homolog leakage, and label quality.
- Random sequence-level splits can overestimate generalization when highly homologous sequences appear across splits. A stronger benchmark should use cluster-aware or family-aware splitting.
- Binary soluble/membrane classification is a simplification of protein localization and topology.

## Next modeling step

A natural extension is a benchmark layer that compares this interpretable baseline against frozen protein-language-model embeddings (for example, ESM-family embeddings) using the same leakage-aware data splits and evaluation harness.

## Data governance

The public repository should contain only synthetic, open, or otherwise redistributable data. No proprietary, confidential, patient, or employer-restricted data should be committed.
