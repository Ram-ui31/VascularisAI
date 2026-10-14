"""Same-fold CAD benchmark: current logistic model vs paper-style DT/Ada+SMOTE+SVM.

The paper names Bayesian optimization but does not provide its search bounds or
fold seed. We use a documented RBF-SVM grid as a reproducible approximation.
All preprocessing, feature ranking, SMOTE, and parameter selection stay within
the training portion of each outer fold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC

from train import pipeline, load_data
from train_v2 import ROOT


def smote(X, y, seed):
    """Minimal numeric SMOTE implementation, invoked on training data only."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    minority = classes[np.argmin(counts)]
    majority_count = int(counts.max())
    minority_rows = X[y == minority]
    need = majority_count - len(minority_rows)
    if need <= 0 or len(minority_rows) < 2:
        return X, y
    from sklearn.neighbors import NearestNeighbors
    k = min(5, len(minority_rows) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(minority_rows)
    neighbors = nn.kneighbors(return_distance=False)[:, 1:]
    base = rng.integers(0, len(minority_rows), size=need)
    neighbor = np.array([rng.choice(neighbors[i]) for i in base])
    step = rng.random((need, 1))
    synthetic = minority_rows[base] + step * (minority_rows[neighbor] - minority_rows[base])
    return np.vstack([X, synthetic]), np.concatenate([y, np.full(need, minority)])


def paper_feature_rank(X, y):
    # Average normalized feature importances from the paper's two selector families.
    dt = ExtraTreesClassifier(n_estimators=160, max_features=None, random_state=42, n_jobs=1)
    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    dt.fit(X, y); ada.fit(X, y)
    d = dt.feature_importances_; a = ada.feature_importances_
    d = d / max(d.sum(), 1e-12); a = a / max(a.sum(), 1e-12)
    return np.argsort((d + a) / 2)[::-1][:29]


def run():
    x, targets = load_data()
    y = targets["CAD"].to_numpy()
    current = pipeline(x, __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(
        C=.35, max_iter=4000, class_weight="balanced", random_state=42))
    outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    p_current = np.zeros(len(y)); p_paper = np.zeros(len(y))
    fold_details = []
    numeric = list(x.select_dtypes(include=np.number).columns)
    categorical = [c for c in x if c not in numeric]
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], sparse_threshold=0)
    for fold, (tr, te) in enumerate(outer.split(x, y), 1):
        p_current[te] = current.fit(x.iloc[tr], y[tr]).predict_proba(x.iloc[te])[:, 1]
        xt = prep.fit_transform(x.iloc[tr]); xv = prep.transform(x.iloc[te])
        selected = paper_feature_rank(xt, y[tr])
        xt, xv = xt[:, selected], xv[:, selected]
        xt_bal, y_bal = smote(xt, y[tr], 900 + fold)
        inner = StratifiedKFold(3, shuffle=True, random_state=200 + fold)
        search = GridSearchCV(
            SVC(kernel="rbf", probability=True, class_weight=None, random_state=42),
            {"C": [.1, 1, 10, 100], "gamma": ["scale", .001, .01, .1]},
            scoring="roc_auc", cv=inner, n_jobs=-1,
        ).fit(xt_bal, y_bal)
        p_paper[te] = search.predict_proba(xv)[:, 1]
        fold_details.append({"fold": fold, "selected_feature_count": len(selected), "best_params": search.best_params_})
        print(f"fold {fold}/10 done", flush=True)

    def score(p):
        pred = p >= .5
        return {k: round(float(v), 4) for k, v in {
            "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p),
        }.items()}
    result = {
        "protocol": "same shuffled stratified 10-fold split (seed 42); training-only preprocessing/SMOTE/feature selection; threshold 0.5",
        "paper_replication_note": "DT/Ada feature ranking + top 29 + training-only SMOTE + RBF-SVM; grid search approximates unspecified Bayesian search bounds",
        "leakage_excluded": ["LAD", "LCX", "RCA", "Cath"],
        "current_model": score(p_current), "paper_style": score(p_paper), "fold_details": fold_details,
    }
    path = ROOT / "reports/benchmark_apples_to_apples.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
