"""Train leakage-safe CAD and vessel classifiers from UCI dataset 411."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/uci_411/extention of Z-Alizadeh sani dataset.xlsx"
TARGETS = {"CAD": "Cath", "LAD": "LAD", "LCX": "LCX", "RCA": "RCA"}
LEAKAGE = ["LAD", "LCX", "RCA", "Cath"]


def load_data() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    raw = pd.read_excel(SOURCE, sheet_name="Sheet 1 - Table 1")
    x = raw.drop(columns=LEAKAGE).copy()
    ys = {
        name: raw[column].map({"Normal": 0, "Stenotic": 1, "CAD": 1}).astype(int)
        for name, column in TARGETS.items()
    }
    return x, ys


def pipeline(x: pd.DataFrame, estimator) -> Pipeline:
    numeric = list(x.select_dtypes(include=np.number).columns)
    categorical = [c for c in x.columns if c not in numeric]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", prep), ("classifier", estimator)])


def scores(y: pd.Series, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    return {k: round(float(v), 4) for k, v in {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
    }.items()}


def main() -> None:
    x, targets = load_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "logistic_regression": LogisticRegression(C=0.35, max_iter=4000, class_weight="balanced", random_state=42),
        "extra_trees": ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1),
    }
    report = {"dataset": {"source": "UCI 411", "rows": len(x), "input_features": x.shape[1], "excluded_for_leakage": LEAKAGE}, "targets": {}}
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    for target, y in targets.items():
        results = {}
        best = None
        best_auc = -1.0
        for name, estimator in candidates.items():
            model = pipeline(x, estimator)
            probability = cross_val_predict(model, x, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
            metric = scores(y, probability)
            results[name] = metric
            if metric["roc_auc"] > best_auc:
                best, best_auc = name, metric["roc_auc"]
        final_model = pipeline(x, candidates[best]).fit(x, y)
        joblib.dump(final_model, ROOT / f"models/{target.lower()}_pipeline.joblib")
        report["targets"][target] = {
            "positive": int(y.sum()), "negative": int((1-y).sum()),
            "selected_model": best, "metrics": results[best], "candidates": results,
        }
        print(target, best, results[best])
    (ROOT / "reports/metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
