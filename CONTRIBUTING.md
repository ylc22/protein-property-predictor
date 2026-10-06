# Contributing

Thanks for helping improve Protein Property Predictor.

## Development setup

1. Create a Python 3.11 environment.
2. Install dependencies:

```bash
pip install -r protein-property-predictor/env/requirements.txt
```

3. Run the test suite before opening a pull request:

```bash
pytest -q
```

## Working on model changes

For changes to features, training, evaluation, or inference:

- keep train/holdout separation intact;
- avoid evaluating on the training set;
- preserve deterministic random seeds where practical;
- update or add tests for new behavior;
- document any new assumptions, metrics, or model limitations.

## Pull request checklist

Before requesting review:

- [ ] Tests pass locally.
- [ ] New behavior has test coverage.
- [ ] README or model-card documentation is updated when behavior changes.
- [ ] No private, proprietary, or patient data is committed.
- [ ] Generated artifacts are excluded unless they are intentionally part of the project.
- [ ] The PR explains the scientific or engineering motivation for the change.

## Scope

This repository is a portfolio and demonstration project. Contributions should improve reproducibility, evaluation quality, usability, or scientific clarity rather than adding complexity without a clear benefit.
