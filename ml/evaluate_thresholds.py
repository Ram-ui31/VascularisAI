"""Nested operating-threshold evaluation for the fixed stacked architecture."""
import json
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from evaluate_stack import stack_pipeline
from train_v2 import ROOT, LEAKAGE, load, metrics


def choose_threshold(y, p):
    candidates=np.linspace(.2,.8,121)
    # Clinical compromise: reward balanced accuracy and F1 equally.
    return max(candidates,key=lambda t:(balanced_accuracy_score(y,p>=t)+f1_score(y,p>=t))/2)


def main():
    x,targets=load(); outer=StratifiedKFold(5,shuffle=True,random_state=42)
    report={"protocol":"outer 5-fold CV; threshold chosen from 3-fold OOF predictions inside each training fold","leakage_excluded":LEAKAGE,"targets":{}}
    for name,y in targets.items():
        probabilities=np.zeros(len(y)); predictions=np.zeros(len(y),dtype=int); thresholds=[]
        for train,test in outer.split(x,y):
            inner=StratifiedKFold(3,shuffle=True,random_state=17)
            inner_p=cross_val_predict(stack_pipeline(x.iloc[train]),x.iloc[train],y.iloc[train],cv=inner,method="predict_proba",n_jobs=-1)[:,1]
            threshold=float(choose_threshold(y.iloc[train],inner_p)); thresholds.append(round(threshold,3))
            model=stack_pipeline(x.iloc[train]).fit(x.iloc[train],y.iloc[train])
            probabilities[test]=model.predict_proba(x.iloc[test])[:,1]
            predictions[test]=(probabilities[test]>=threshold).astype(int)
        result=metrics(y,probabilities)
        result["thresholded_accuracy"]=round(float(accuracy_score(y,predictions)),4)
        result["thresholded_f1"]=round(float(f1_score(y,predictions)),4)
        result["fold_thresholds"]=thresholds
        report["targets"][name]=result; print(name,result,flush=True)
    (ROOT/"reports/metrics_thresholds.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__": main()
