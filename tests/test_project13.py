import os
import json
import pytest
import pandas as pd

from project13_predictive_app.data import load_data, get_feature_target
from project13_predictive_app.predict import predict_from_dict


def test_load_data_shape():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] >= 1


def test_get_feature_target():
    df = load_data()
    X, y = get_feature_target(df)
    assert "Purchase Amount (USD)" not in X.columns
    assert len(y) == len(X)


def test_artifacts_exist():
    base = os.path.join(os.path.dirname(__file__), "..")
    art_dir = os.path.abspath(os.path.join(base, "project13_artifacts"))
    assert os.path.isdir(art_dir)
    assert os.path.isfile(os.path.join(art_dir, "model.joblib"))
    assert os.path.isfile(os.path.join(art_dir, "preprocessor.joblib"))
    assert os.path.isfile(os.path.join(art_dir, "metrics.json"))


def test_predict_returns_number():
    example = {
        "Age": 34,
        "Gender": "Male",
        "Category": "Clothing",
        "Location": "CA",
        "Color": "Blue",
        "Season": "Spring",
        "Review Rating": 4,
        "Subscription Status": "Yes",
        "Shipping Type": "Express",
        "Discount Applied": "No",
        "Promo Code Used": "No",
        "Previous Purchases": 5,
        "Payment Method": "Credit Card",
        "Frequency of Purchases": "Monthly",
    }
    val = predict_from_dict(example)
    assert isinstance(val, float) or isinstance(val, int)
