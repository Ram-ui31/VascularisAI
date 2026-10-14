"""Shared 10-fold comparison: CatBoost vs paper-style 17-feature GA stack."""
import json
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from genetic_vessel_stack import preprocessor, genetic_select, stack
from train_v2 import ROOT, LEAKAGE, load, metrics

CONFIGS = {
    "LAD": {"depth": 3, "learning_rate": .03, "l2_leaf_reg": 5},
    "LCX": {"depth": 4, "learning_rate": .03, "l2_leaf_reg": 10},
    "RCA": {"depth": 6, "learning_rate": .02, "l2_leaf_reg": 20},
}

def main():
    x, ys = load()
    outer = StratifiedKFold(10, shuffle=True, random_state=42)
    report = {"protocol": "shared shuffled stratified 10-fold CV seed 42; threshold 0.5; fold-contained training", "leakage_excluded": LEAKAGE, "targets": {}}
    for target in ["LAD", "LCX", "RCA"]:
        y = ys[target].to_numpy(); raw = x.copy()
        cats = list(raw.select_dtypes(exclude=np.number).columns); raw[cats] = raw[cats].astype(str)
        catidx = [raw.columns.get_loc(c) for c in cats]
        p_cat = np.zeros(len(y)); p_ga = np.zeros(len(y)); selected = []
        for fold, (tr, te) in enumerate(outer.split(x, y), 1):
            cm = CatBoostClassifier(iterations=900, eval_metric="AUC", auto_class_weights="Balanced", verbose=False, random_seed=42, allow_writing_files=False, **CONFIGS[target])
            cm.fit(raw.iloc[tr], y[tr], cat_features=catidx)
            p_cat[te] = cm.predict_proba(raw.iloc[te])[:, 1]
            prep, ordered = preprocessor(x.iloc[tr]); xt = prep.fit_transform(x.iloc[tr]); xv = prep.transform(x.iloc[te])
            mask, _ = genetic_select(xt, y[tr], seed=100+fold, population_size=18, generations=8)
            gm = stack(100+fold).fit(xt[:, mask], y[tr])
            p_ga[te] = gm.predict_proba(xv[:, mask])[:, 1]
            selected.append([ordered[i] for i in np.flatnonzero(mask)])
            print(target, f"fold {fold}/10", flush=True)
        report["targets"][target] = {"catboost": metrics(y, p_cat), "paper_style_genetic_stack": metrics(y, p_ga), "selected_features_by_fold": selected}
        print(target, report["targets"][target]["catboost"], report["targets"][target]["paper_style_genetic_stack"], flush=True)
    (ROOT / "reports/benchmark_vessels_10fold.json").write_text(json.dumps(report, indent=2))

if __name__ == "__main__": main()
