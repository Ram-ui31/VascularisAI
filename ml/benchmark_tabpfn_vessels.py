"""TabPFN pretrained transformer evaluation on the shared 10-fold vessel splits."""
import argparse
import json
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from tabpfn import TabPFNClassifier
from train_v2 import ROOT, LEAKAGE, load, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', default='LAD,LCX,RCA')
    parser.add_argument('--output', default='benchmark_tabpfn_vessels.json')
    args = parser.parse_args()
    x, ys = load(); outer = StratifiedKFold(10, shuffle=True, random_state=42)
    nums = list(x.select_dtypes(include=np.number).columns); cats = [c for c in x if c not in nums]
    prep = ColumnTransformer([
        ('num', Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]), nums),
        ('cat', Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('enc',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]), cats),
    ], sparse_threshold=0)
    report={'protocol':'shared shuffled stratified 10-fold CV seed 42; TabPFN pretrained; fold-contained imputation/encoding','leakage_excluded':LEAKAGE,'targets':{}}
    for target in args.targets.split(','):
        y=ys[target].to_numpy(); p=np.zeros(len(y))
        for fold,(tr,te) in enumerate(outer.split(x,y),1):
            xt=prep.fit_transform(x.iloc[tr]); xv=prep.transform(x.iloc[te])
            model=TabPFNClassifier(n_estimators=8,device='auto',random_state=42,show_progress_bar=False,enable_torch_compile=False)
            model.fit(xt,y[tr]); p[te]=model.predict_proba(xv)[:,1]
            print(target, f'fold {fold}/10', flush=True)
        report['targets'][target]=metrics(y,p); print(target,report['targets'][target],flush=True)
    (ROOT/'reports'/args.output).write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()
