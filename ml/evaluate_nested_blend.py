"""Leakage-safe nested blending of the strongest vessel-model families.

Blend weights and the classification threshold are selected using only inner
training-fold predictions. The outer fold remains untouched until evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from evaluate_stack import stack_pipeline
from blended_model import VesselBlend
from train import pipeline
from train_v2 import LEAKAGE, ROOT, load, metrics


CAT_CONFIG = {
    "LAD": {"depth": 3, "learning_rate": .03, "l2_leaf_reg": 5},
    "LCX": {"depth": 4, "learning_rate": .03, "l2_leaf_reg": 10},
    "RCA": {"depth": 6, "learning_rate": .02, "l2_leaf_reg": 20},
}


def cat_data(x):
    out = x.copy()
    cats = list(out.select_dtypes(exclude=np.number).columns)
    out[cats] = out[cats].astype(str)
    return out, [out.columns.get_loc(c) for c in cats]


def cat_model(target):
    return CatBoostClassifier(
        iterations=900, eval_metric="AUC", auto_class_weights="Balanced",
        verbose=False, random_seed=42, allow_writing_files=False,
        **CAT_CONFIG[target],
    )


def tree_model(x):
    return pipeline(x, ExtraTreesClassifier(
        n_estimators=600, max_depth=8, min_samples_leaf=3,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    ))


def choose_blend(y, probabilities):
    best = (-1.0, None, None)
    # Simplex grid for [ExtraTrees, CatBoost, heterogeneous stack].
    for a in np.arange(0, 1.01, .1):
        for b in np.arange(0, 1.01 - a, .1):
            w = np.array([a, b, 1 - a - b])
            p = probabilities @ w
            for threshold in np.arange(.35, .651, .025):
                score = accuracy_score(y, p >= threshold)
                # Prefer a threshold near 0.5 when scores tie.
                candidate = (score, -abs(threshold - .5))
                incumbent = (best[0], -abs(best[2] - .5)) if best[1] is not None else (-1, -1)
                if candidate > incumbent:
                    best = (score, w.copy(), float(threshold))
    return best[1], best[2], best[0]


def inner_predictions(x, y, target, cv):
    tree = tree_model(x)
    p_tree = cross_val_predict(tree, x, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

    xc, cats = cat_data(x)
    p_cat = np.zeros(len(y))
    for tr, va in cv.split(xc, y):
        model = cat_model(target).fit(xc.iloc[tr], y.iloc[tr], cat_features=cats)
        p_cat[va] = model.predict_proba(xc.iloc[va])[:, 1]

    stack = stack_pipeline(x)
    p_stack = cross_val_predict(stack, x, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return np.column_stack([p_tree, p_cat, p_stack])


def outer_predictions(x, y, target):
    outer = StratifiedKFold(5, shuffle=True, random_state=42)
    probabilities = np.zeros(len(y)); predictions = np.zeros(len(y), dtype=int)
    selections = []
    for fold, (tr, te) in enumerate(outer.split(x, y), 1):
        xt, xv, yt = x.iloc[tr], x.iloc[te], y.iloc[tr]
        inner = StratifiedKFold(3, shuffle=True, random_state=100 + fold)
        inner_p = inner_predictions(xt, yt, target, inner)
        weights, threshold, inner_accuracy = choose_blend(yt.to_numpy(), inner_p)

        tree = tree_model(xt).fit(xt, yt)
        p_tree = tree.predict_proba(xv)[:, 1]
        xct, cats = cat_data(xt); xcv, _ = cat_data(xv)
        cat = cat_model(target).fit(xct, yt, cat_features=cats)
        p_cat = cat.predict_proba(xcv)[:, 1]
        stack = stack_pipeline(xt).fit(xt, yt)
        p_stack = stack.predict_proba(xv)[:, 1]
        p = np.column_stack([p_tree, p_cat, p_stack]) @ weights
        probabilities[te] = p
        predictions[te] = p >= threshold
        selections.append({
            "fold": fold, "weights": dict(zip(["extra_trees", "catboost", "stack"], weights.round(3))),
            "threshold": round(threshold, 3), "inner_accuracy": round(float(inner_accuracy), 4),
        })
    result = metrics(y, probabilities)
    result["accuracy_nested_threshold"] = round(float(accuracy_score(y, predictions)), 4)
    return result, selections


def fit_final(x, y, target, selections):
    weights = np.median(np.array([
        list(s["weights"].values()) for s in selections
    ]), axis=0)
    weights = weights / weights.sum()
    threshold = float(np.median([s["threshold"] for s in selections]))
    tree = tree_model(x).fit(x, y)
    xc, cats = cat_data(x)
    cat = cat_model(target).fit(xc, y, cat_features=cats)
    stack = stack_pipeline(x).fit(x, y)
    return VesselBlend(tree, cat, stack, xc.columns[cats], weights, threshold)


def main():
    x, targets = load()
    report = {
        "protocol": "5-fold untouched outer CV; blend weights and threshold selected by 3-fold inner CV",
        "leakage_excluded": LEAKAGE,
        "targets": {},
    }
    for target in ["LAD", "LCX", "RCA"]:
        score, selections = outer_predictions(x, targets[target], target)
        report["targets"][target] = {"metrics": score, "fold_selections": selections}
        if target in {"LAD", "RCA"}:
            model = fit_final(x, targets[target], target, selections)
            outdir = Path(ROOT) / "models/blend"
            outdir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, outdir / f"{target.lower()}_pipeline.joblib")
        print(target, score, flush=True)
    path = Path(ROOT) / "reports/metrics_nested_blend.json"
    path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
