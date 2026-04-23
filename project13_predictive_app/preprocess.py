from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd


def build_preprocessor(X: pd.DataFrame):
    """Create a ColumnTransformer: scale numeric, one-hot encode categorical."""
    numeric_features = [c for c in ["Age", "Review Rating", "Previous Purchases"] if c in X.columns]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    # use sparse_output=False for newer scikit-learn versions
    categorical_transformer = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ], remainder="drop")

    return preprocessor
