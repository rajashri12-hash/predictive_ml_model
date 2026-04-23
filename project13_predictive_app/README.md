# Project 13 — Predictive Analytics Application

This small app trains a regression model to predict `Purchase Amount (USD)` from the provided shopping dataset and exposes a Streamlit app for interactive predictions.

Quick steps:

1. Install dependencies:

```bash
pip install -r project13_predictive_app/requirements.txt
```

2. Train and save model artifacts:

```bash
python -m project13_predictive_app.train
```

3. Run the Streamlit app:

```bash
streamlit run project13_predictive_app/app.py
```

Artifacts are saved to `project13_artifacts/` inside the workspace root after training.

Tests
-----

Run unit tests with `pytest` from the repository root:

```bash
pip install pytest
pytest -q
```

The tests validate data loading, the preprocessing/trained artifacts, and a sample prediction.
