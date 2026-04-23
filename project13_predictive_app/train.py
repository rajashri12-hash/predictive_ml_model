"""Train models and save the best regressor and preprocessor."""
from pathlib import Path
import joblib
import json

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from .data import load_data, get_feature_target
from .preprocess import build_preprocessor


def train_and_save(csv_path: str = None, out_dir: str = None):
    df = load_data(csv_path)
    X, y = get_feature_target(df)

    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit preprocessor
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_t, y_train)
        preds = model.predict(X_test_t)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        results[name] = {"rmse": rmse}

    # choose best
    best_name = min(results.keys(), key=lambda k: results[k]["rmse"])
    best_model = models[best_name]

    out = Path(out_dir or Path(__file__).resolve().parents[1]) / "project13_artifacts"
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.joblib"
    preproc_path = out / "preprocessor.joblib"
    metrics_path = out / "metrics.json"

    joblib.dump(best_model, model_path)
    joblib.dump(preprocessor, preproc_path)
    with open(metrics_path, "w") as f:
        json.dump({"results": results, "best": best_name}, f, indent=2)

    return {"model_path": str(model_path), "preprocessor_path": str(preproc_path), "metrics": results, "best": best_name}


if __name__ == "__main__":
    print("Training model...")
    out = train_and_save()
    print("Saved:", out)
