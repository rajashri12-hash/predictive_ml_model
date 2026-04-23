from pathlib import Path
import joblib
import pandas as pd


def _resolve_artifacts_dir(artifacts_dir: str = None) -> Path:
    """Find artifact directory in common run contexts (local/cloud)."""
    candidates = []
    if artifacts_dir:
        candidates.append(Path(artifacts_dir))

    # 1) current working directory (Streamlit Cloud usually runs from repo root)
    candidates.append(Path.cwd() / "project13_artifacts")
    # 2) package-relative repo root
    candidates.append(Path(__file__).resolve().parents[1] / "project13_artifacts")

    required = ["model.joblib", "preprocessor.joblib", "classifier.joblib", "classifier_preprocessor.joblib"]
    for base in candidates:
        if base.exists() and all((base / f).exists() for f in required):
            return base

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find artifact files. Checked:\n"
        f"{searched}\n"
        "Expected files: model.joblib, preprocessor.joblib, classifier.joblib, classifier_preprocessor.joblib"
    )


def load_artifacts(artifacts_dir: str = None):
    base = _resolve_artifacts_dir(artifacts_dir)
    model = joblib.load(base / "model.joblib")
    preprocessor = joblib.load(base / "preprocessor.joblib")
    return preprocessor, model


def predict_from_dict(d: dict, artifacts_dir: str = None):
    preprocessor, model = load_artifacts(artifacts_dir)
    df = pd.DataFrame([d])
    X_t = preprocessor.transform(df)
    pred = model.predict(X_t)
    return float(pred[0])


def load_classifier_artifacts(artifacts_dir: str = None):
    base = _resolve_artifacts_dir(artifacts_dir)
    clf = joblib.load(base / "classifier.joblib")
    preproc = joblib.load(base / "classifier_preprocessor.joblib")
    return preproc, clf


def predict_subscription_from_dict(d: dict, artifacts_dir: str = None):
    preproc, clf = load_classifier_artifacts(artifacts_dir)
    df = pd.DataFrame([d])
    X_t = preproc.transform(df)
    prob = float(clf.predict_proba(X_t)[0, 1])
    label = int(clf.predict(X_t)[0])
    return {'probability': prob, 'label': 'Yes' if label == 1 else 'No'}


if __name__ == "__main__":
    example = {
        "Age": 30,
        "Gender": "Female",
        "Category": "Accessories",
        "Location": "NY",
        "Color": "Red",
        "Season": "Summer",
        "Review Rating": 4,
        "Subscription Status": "Yes",
        "Shipping Type": "Standard",
        "Discount Applied": "No",
        "Promo Code Used": "No",
        "Previous Purchases": 5,
        "Payment Method": "Credit Card",
        "Frequency of Purchases": "Monthly",
    }
    print(predict_from_dict(example))
