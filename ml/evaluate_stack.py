"""Evaluate a fixed heterogeneous stack with untouched five-fold predictions."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold

from train_v2 import ROOT, LEAKAGE, load, metrics, search


def stack_pipeline(x):
    # Reuse the fitted-shape-safe preprocessor definition from the search pipeline.
    preprocess = search(x).estimator.named_steps["preprocess"]
    estimators = [
        ("linear", LogisticRegression(C=.1, max_iter=5000, class_weight="balanced", random_state=42)),
        ("rbf", SVC(C=2, gamma=.01, probability=True, class_weight="balanced", random_state=42)),
        ("trees", ExtraTreesClassifier(n_estimators=600, max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)),
        ("boost", HistGradientBoostingClassifier(max_leaf_nodes=7, learning_rate=.04, l2_regularization=1, random_state=42)),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=.2, class_weight="balanced", random_state=42),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("variance", VarianceThreshold()), ("select", SelectKBest(f_classif, k=35)), ("classifier", stack)])


def main():
    x, targets = load()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    report = {"protocol": "fixed heterogeneous stack, 5-fold stratified out-of-fold", "leakage_excluded": LEAKAGE, "targets": {}}
    outdir = ROOT / "models/stack"
    outdir.mkdir(parents=True, exist_ok=True)
    for name, y in targets.items():
        model = stack_pipeline(x)
        probability = cross_val_predict(model, x, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        report["targets"][name] = metrics(y, probability)
        model.fit(x, y)
        joblib.dump(model, outdir / f"{name.lower()}_pipeline.joblib")
        print(name, report["targets"][name])
    (ROOT / "reports/metrics_stack.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
