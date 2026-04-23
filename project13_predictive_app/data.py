import pandas as pd
from pathlib import Path


def load_data(csv_path: str = None) -> pd.DataFrame:
    """Load dataset from path or default CSV in workspace."""
    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[1] / "Shopping_Trends_And_Customer_Behaviour_Dataset.csv"
    return pd.read_csv(csv_path)


def get_feature_target(df: pd.DataFrame):
    """Return X, y for modeling. Drops identifiers and target-prep."""
    df = df.copy()
    y = df["Purchase Amount (USD)"].values
    drop_cols = ["Customer ID", "Item Purchased", "Purchase Amount (USD)"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y
