from pathlib import Path
import joblib
import pandas as pd


def load_artifacts(artifacts_dir: str = None):
    base = Path(artifacts_dir) if artifacts_dir else Path(__file__).resolve().parents[1] / "project13_artifacts"
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
    base = Path(artifacts_dir) if artifacts_dir else Path(__file__).resolve().parents[1] / "project13_artifacts"
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
