"""Train and save a Subscription Status classifier."""
from pathlib import Path
import joblib
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from .data import load_data, get_feature_target
from .preprocess import build_preprocessor


def train_and_save_classifier(csv_path: str = None, out_dir: str = None):
    df = load_data(csv_path)
    df = df.copy()
    # target: Subscription Status -> binary
    y = df['Subscription Status'].map({'Yes': 1, 'No': 0}).values
    drop_cols = ['Customer ID', 'Item Purchased', 'Subscription Status', 'Purchase Amount (USD)']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_t, y_train)
        preds = model.predict(X_test_t)
        probs = model.predict_proba(X_test_t)[:, 1]
        results[name] = {
            'accuracy': float(accuracy_score(y_test, preds)),
            'precision': float(precision_score(y_test, preds)),
            'recall': float(recall_score(y_test, preds)),
            'f1': float(f1_score(y_test, preds)),
            'roc_auc': float(roc_auc_score(y_test, probs)),
        }

    best_name = max(results.keys(), key=lambda k: results[k]['f1'])
    best_model = models[best_name]

    out = Path(out_dir or Path(__file__).resolve().parents[1]) / 'project13_artifacts'
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / 'classifier.joblib'
    preproc_path = out / 'classifier_preprocessor.joblib'
    metrics_path = out / 'classifier_metrics.json'

    joblib.dump(best_model, model_path)
    joblib.dump(preprocessor, preproc_path)
    with open(metrics_path, 'w') as f:
        json.dump({'results': results, 'best': best_name}, f, indent=2)

    return {'model_path': str(model_path), 'preprocessor_path': str(preproc_path), 'metrics': results, 'best': best_name}


if __name__ == '__main__':
    print('Training classifier...')
    out = train_and_save_classifier()
    print('Saved classifier artifacts:', out)
