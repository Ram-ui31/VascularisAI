"""Leakage-safe nested-CV benchmark and tuned Vascularis model trainer.

All preprocessing, feature selection and hyperparameter tuning happen inside
each training fold. The untouched outer fold is used only for evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/uci_411/extention of Z-Alizadeh sani dataset.xlsx"
TARGETS = {"CAD": "Cath", "LAD": "LAD", "LCX": "LCX", "RCA": "RCA"}
LEAKAGE = ["LAD", "LCX", "RCA", "Cath"]


def load():
    df = pd.read_excel(SOURCE, sheet_name="Sheet 1 - Table 1")
    x = df.drop(columns=LEAKAGE)
    y = {name: df[col].map({"Normal": 0, "Stenotic": 1, "CAD": 1}).astype(int) for name, col in TARGETS.items()}
    return x, y


def search(x: pd.DataFrame) -> GridSearchCV:
    numeric = list(x.select_dtypes(include=np.number).columns)
    categorical = [c for c in x if c not in numeric]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], sparse_threshold=0)
    pipe = Pipeline([
        ("preprocess", prep),
        ("variance", VarianceThreshold()),
        ("select", SelectKBest(score_func=f_classif)),
        ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
    ])
    grids = [
        {
            "select__k": [20, 35, "all"],
            "classifier": [LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)],
            "classifier__C": [.1, 1, 10],
        },
        {
            "select__k": [20, 35, "all"],
            "classifier": [SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)],
            "classifier__C": [.5, 2, 10, 50],
            "classifier__gamma": ["scale", .01, .05],
        },
        {
            "select__k": [35, "all"],
            "classifier": [ExtraTreesClassifier(n_estimators=600, class_weight="balanced", random_state=42, n_jobs=-1)],
            "classifier__max_depth": [None, 8],
            "classifier__min_samples_leaf": [1, 3, 6],
        },
        {
            "select__k": [20, 35, "all"],
            "classifier": [HistGradientBoostingClassifier(random_state=42)],
            "classifier__learning_rate": [.04, .1],
            "classifier__max_leaf_nodes": [7, 15],
            "classifier__l2_regularization": [0, 1],
        },
    ]
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
    return GridSearchCV(pipe, grids, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)


def metrics(y, p):
    pred = (p >= .5).astype(int)
    return {k: round(float(v), 4) for k, v in {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
    }.items()}


def compact_params(params):
    clean = {}
    for key, value in params.items():
        if key == "classifier":
            clean["classifier"] = type(value).__name__
        elif isinstance(value, (str, int, float)) or value is None:
            clean[key] = value
    return clean


def main():
    x, targets = load()
    out = {"protocol": "5-fold outer / 3-fold inner nested stratified CV", "leakage_excluded": LEAKAGE, "targets": {}}
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_dir = ROOT / "models/v2"
    model_dir.mkdir(parents=True, exist_ok=True)
    for target, y in targets.items():
        oof = np.zeros(len(y))
        folds = []
        for fold, (train, test) in enumerate(outer.split(x, y), 1):
            tuned = search(x.iloc[train])
            tuned.fit(x.iloc[train], y.iloc[train])
            oof[test] = tuned.predict_proba(x.iloc[test])[:, 1]
            folds.append({"fold": fold, "inner_auc": round(float(tuned.best_score_), 4), "best": compact_params(tuned.best_params_)})
        final = search(x)
        final.fit(x, y)
        joblib.dump(final.best_estimator_, model_dir / f"{target.lower()}_pipeline.joblib")
        out["targets"][target] = {
            "metrics": metrics(y, oof),
            "selected_final": compact_params(final.best_params_),
            "fold_selections": folds,
        }
        print(target, out["targets"][target]["metrics"], out["targets"][target]["selected_final"])
    (ROOT / "reports/metrics_v2.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
