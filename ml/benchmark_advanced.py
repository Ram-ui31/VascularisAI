"""Leakage-safe advanced benchmarks: CatBoost, XGBoost and TabPFN.

Use --models catboost,xgboost or --models tabpfn. Every reported prediction is
made for an untouched outer fold. Hyperparameter choice stays inside that fold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from train_v2 import ROOT, LEAKAGE, load, metrics


def prepared_raw(x: pd.DataFrame):
    out = x.copy()
    cats = list(out.select_dtypes(exclude=np.number).columns)
    out[cats] = out[cats].astype(str)
    return out, [out.columns.get_loc(c) for c in cats]


def catboost_nested(x, y, outer):
    x, cats = prepared_raw(x)
    configs = [
        {"depth": 3, "learning_rate": .03, "l2_leaf_reg": 5},
        {"depth": 4, "learning_rate": .03, "l2_leaf_reg": 10},
        {"depth": 5, "learning_rate": .02, "l2_leaf_reg": 10},
        {"depth": 6, "learning_rate": .02, "l2_leaf_reg": 20},
    ]
    inner = StratifiedKFold(3, shuffle=True, random_state=17)
    oof = np.zeros(len(y)); selected = []
    for train, test in outer.split(x, y):
        best_auc, best_cfg = -1, None
        for cfg in configs:
            inner_p = np.zeros(len(train))
            for a, b in inner.split(x.iloc[train], y.iloc[train]):
                ia, ib = train[a], train[b]
                model = CatBoostClassifier(iterations=600, eval_metric="AUC", auto_class_weights="Balanced", verbose=False, random_seed=42, allow_writing_files=False, **cfg)
                model.fit(x.iloc[ia], y.iloc[ia], cat_features=cats)
                inner_p[b] = model.predict_proba(x.iloc[ib])[:, 1]
            auc = metrics(y.iloc[train], inner_p)["roc_auc"]
            if auc > best_auc: best_auc, best_cfg = auc, cfg
        model = CatBoostClassifier(iterations=900, eval_metric="AUC", auto_class_weights="Balanced", verbose=False, random_seed=42, allow_writing_files=False, **best_cfg)
        model.fit(x.iloc[train], y.iloc[train], cat_features=cats)
        oof[test] = model.predict_proba(x.iloc[test])[:, 1]
        selected.append(best_cfg)
    final_cfg = max(configs, key=lambda c: sum(c == s for s in selected))
    final = CatBoostClassifier(iterations=900, eval_metric="AUC", auto_class_weights="Balanced", verbose=False, random_seed=42, allow_writing_files=False, **final_cfg)
    final.fit(x, y, cat_features=cats)
    return metrics(y, oof), final, {"outer_selections": selected, "final": final_cfg}


def xgb_search(x):
    numeric = list(x.select_dtypes(include=np.number).columns)
    categorical = [c for c in x if c not in numeric]
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    pipe = Pipeline([("preprocess", prep), ("classifier", XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=1, subsample=.85, colsample_bytree=.8))])
    grid = {"classifier__n_estimators": [250, 500], "classifier__max_depth": [2, 3], "classifier__learning_rate": [.03, .07], "classifier__min_child_weight": [3, 8], "classifier__reg_lambda": [3, 12]}
    return GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=17), scoring="roc_auc", n_jobs=-1, refit=True)


def xgboost_nested(x, y, outer):
    oof = np.zeros(len(y)); selected=[]
    for train, test in outer.split(x, y):
        tuned=xgb_search(x.iloc[train]); tuned.fit(x.iloc[train],y.iloc[train])
        oof[test]=tuned.predict_proba(x.iloc[test])[:,1]; selected.append(tuned.best_params_)
    final=xgb_search(x); final.fit(x,y)
    return metrics(y,oof), final.best_estimator_, {"outer_selections":selected,"final":final.best_params_}


def tabpfn_cv(x, y, outer):
    from tabpfn import TabPFNClassifier
    numeric = list(x.select_dtypes(include=np.number).columns)
    categorical = [c for c in x if c not in numeric]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
        ("cat", Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))]),categorical),
    ], sparse_threshold=0)
    oof=np.zeros(len(y))
    for train,test in outer.split(x,y):
        xt=prep.fit_transform(x.iloc[train]); xv=prep.transform(x.iloc[test])
        model=TabPFNClassifier(n_estimators=8,device="auto",random_state=42,show_progress_bar=False)
        model.fit(xt,y.iloc[train].to_numpy()); oof[test]=model.predict_proba(xv)[:,1]
    final_prep=prep.fit(x); final=TabPFNClassifier(n_estimators=8,device="auto",random_state=42,show_progress_bar=False)
    final.fit(final_prep.transform(x),y.to_numpy())
    return metrics(y,oof), {"preprocess":final_prep,"classifier":final}, {"n_estimators":8}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--models",default="catboost,xgboost")
    wanted=parser.parse_args().models.split(",")
    x,targets=load(); outer=StratifiedKFold(5,shuffle=True,random_state=42)
    report={"protocol":"5-fold untouched outer CV; 3-fold inner tuning where applicable","leakage_excluded":LEAKAGE,"models":{}}
    outdir=ROOT/"models/advanced"; outdir.mkdir(parents=True,exist_ok=True)
    runners={"catboost":catboost_nested,"xgboost":xgboost_nested,"tabpfn":tabpfn_cv}
    for model_name in wanted:
        report["models"][model_name]={}
        for target,y in targets.items():
            score,model,details=runners[model_name](x,y,outer)
            report["models"][model_name][target]={"metrics":score,"details":details}
            joblib.dump(model,outdir/f"{model_name}_{target.lower()}.joblib")
            print(model_name,target,score,flush=True)
    path=ROOT/f"reports/advanced_{'_'.join(wanted)}.json"; path.write_text(json.dumps(report,indent=2,default=str))


if __name__=="__main__": main()
