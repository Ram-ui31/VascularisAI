"""Paper-style artery-specific 17-feature genetic wrapper + stacked ensemble.

The genetic search is rerun inside every outer training fold. It never sees the
outer evaluation patients. All four angiography-derived fields remain excluded.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from xgboost import XGBClassifier

from train_v2 import ROOT, LEAKAGE, load, metrics

N_FEATURES = 17


def preprocessor(x):
    numeric=list(x.select_dtypes(include=np.number).columns)
    categorical=[c for c in x if c not in numeric]
    return ColumnTransformer([
        ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))]),categorical),
    ],sparse_threshold=0), numeric+categorical


def mask_from_indices(n, indices):
    mask=np.zeros(n,dtype=bool); mask[indices]=True; return mask


def genetic_select(x, y, seed=42, population_size=22, generations=10):
    rng=np.random.default_rng(seed); n=x.shape[1]
    cv=StratifiedKFold(3,shuffle=True,random_state=seed)
    cache={}
    def fitness(mask):
        key=tuple(np.flatnonzero(mask))
        if key not in cache:
            clf=SVC(C=2,gamma="scale",kernel="rbf",class_weight="balanced",probability=True,random_state=seed)
            p=cross_val_predict(clf,x[:,mask],y,cv=cv,method="predict_proba",n_jobs=1)[:,1]
            cache[key]=roc_auc_score(y,p)-0.0002*abs(mask.sum()-N_FEATURES)
        return cache[key]
    population=[mask_from_indices(n,rng.choice(n,N_FEATURES,replace=False)) for _ in range(population_size)]
    history=[]
    for generation in range(generations):
        ranked=sorted(((fitness(m),m) for m in population),key=lambda z:z[0],reverse=True)
        history.append(round(float(ranked[0][0]),4)); elites=[m.copy() for _,m in ranked[:6]]
        children=elites.copy()
        while len(children)<population_size:
            a,b=rng.choice(elites,2,replace=False); child=np.where(rng.random(n)<.5,a,b)
            on=np.flatnonzero(child); off=np.flatnonzero(~child)
            if len(on)>N_FEATURES: child[rng.choice(on,len(on)-N_FEATURES,replace=False)]=False
            elif len(on)<N_FEATURES: child[rng.choice(off,N_FEATURES-len(on),replace=False)]=True
            if rng.random()<.45:
                on=np.flatnonzero(child); off=np.flatnonzero(~child)
                child[rng.choice(on)]=False; child[rng.choice(off)]=True
            children.append(child)
        population=children
    ranked=sorted(((fitness(m),m) for m in population),key=lambda z:z[0],reverse=True)
    return ranked[0][1],history


def stack(seed=42):
    base=[
        ("svm",SVC(C=2,kernel="rbf",gamma="scale",probability=True,class_weight="balanced",random_state=seed)),
        ("gbdt",GradientBoostingClassifier(n_estimators=180,max_depth=2,learning_rate=.035,subsample=.85,random_state=seed)),
        ("xgb",XGBClassifier(n_estimators=300,max_depth=2,learning_rate=.035,subsample=.85,colsample_bytree=.9,min_child_weight=5,reg_lambda=8,eval_metric="logloss",n_jobs=1,random_state=seed)),
        ("ada",AdaBoostClassifier(n_estimators=220,learning_rate=.035,random_state=seed)),
    ]
    return StackingClassifier(estimators=base,final_estimator=LogisticRegression(C=.2,class_weight="balanced",max_iter=3000,random_state=seed),stack_method="predict_proba",cv=5,n_jobs=-1)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--targets",default="LAD,LCX,RCA")
    names=parser.parse_args().targets.split(",")
    raw,ys=load(); outer=StratifiedKFold(5,shuffle=True,random_state=42)
    report={"protocol":"17-feature GA wrapper inside each outer fold; SVM/GBDT/XGBoost/AdaBoost -> LR stack","leakage_excluded":LEAKAGE,"targets":{}}
    outdir=ROOT/"models/genetic_stack"; outdir.mkdir(parents=True,exist_ok=True)
    for name in names:
        y=ys[name]; p=np.zeros(len(y)); fold_features=[]; histories=[]
        for fold,(train,test) in enumerate(outer.split(raw,y),1):
            prep,ordered=preprocessor(raw.iloc[train]); xt=prep.fit_transform(raw.iloc[train]); xv=prep.transform(raw.iloc[test])
            mask,history=genetic_select(xt,y.iloc[train].to_numpy(),seed=100+fold)
            model=stack(100+fold).fit(xt[:,mask],y.iloc[train]); p[test]=model.predict_proba(xv[:,mask])[:,1]
            fold_features.append([ordered[i] for i in np.flatnonzero(mask)]); histories.append(history)
        prep,ordered=preprocessor(raw); full=prep.fit_transform(raw)
        mask,history=genetic_select(full,y.to_numpy(),seed=999,population_size=28,generations=14)
        model=stack(999).fit(full[:,mask],y)
        artifact={"preprocess":prep,"feature_mask":mask,"feature_names":[ordered[i] for i in np.flatnonzero(mask)],"classifier":model}
        joblib.dump(artifact,outdir/f"{name.lower()}_pipeline.joblib")
        report["targets"][name]={"metrics":metrics(y,p),"outer_features":fold_features,"final_features":artifact["feature_names"],"fitness_history":histories,"final_history":history}
        print(name,report["targets"][name]["metrics"],artifact["feature_names"],flush=True)
    (ROOT/"reports/metrics_genetic_stack.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__": main()
