"""Local prediction API for the four trained Vascularis pipelines."""
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "ml"))
RAW = ROOT / "data/raw/uci_411/extention of Z-Alizadeh sani dataset.xlsx"
TARGET_COLUMNS = ["LAD", "LCX", "RCA", "Cath"]
raw = pd.read_excel(RAW, sheet_name="Sheet 1 - Table 1").drop(columns=TARGET_COLUMNS)
defaults = {c: (float(raw[c].median()) if pd.api.types.is_numeric_dtype(raw[c]) else str(raw[c].mode().iloc[0])) for c in raw.columns}
MODEL_PATHS = {
    "CAD": ROOT / "models/cad_pipeline.joblib",
    "LAD": ROOT / "models/blend/lad_pipeline.joblib",
    "LCX": ROOT / "models/advanced/catboost_lcx.joblib",
    "RCA": ROOT / "models/blend/rca_pipeline.joblib",
}
models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}

app = FastAPI(title="Vascularis AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

class Patient(BaseModel):
    features: dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok", "targets": list(models), "input_features": len(defaults)}

@app.get("/schema")
def schema():
    return {"features": defaults, "choices": {c: sorted(raw[c].dropna().unique().tolist()) for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c]) or raw[c].nunique() <= 5}, "targets_excluded": TARGET_COLUMNS}

@app.post("/predict")
def predict(patient: Patient):
    unknown = sorted(set(patient.features) - set(defaults))
    if unknown:
        raise HTTPException(422, detail="Unknown clinical fields: " + ", ".join(unknown))
    if not patient.features:
        raise HTTPException(422, detail="Enter at least one clinical measurement.")
    if patient.features.get("Sex") == "Female":
        patient.features["Sex"] = "Fmale"
    for field, value in patient.features.items():
        if pd.api.types.is_numeric_dtype(raw[field]):
            if not isinstance(value, (float, int)) or not np.isfinite(value):
                raise HTTPException(422, detail=f"{field} must be a finite number.")
            bounds = {"Age": (18, 110), "BP": (50, 260), "LDL": (0, 600), "EF-TTE": (0, 100), "BMI": (10, 80)}
            if field in bounds and not bounds[field][0] <= value <= bounds[field][1]:
                raise HTTPException(422, detail=f"Check {field}: expected {bounds[field][0]}–{bounds[field][1]} in the displayed units.")
            if raw[field].nunique() <= 5 and value not in raw[field].unique():
                raise HTTPException(422, detail=f"Choose a valid value for {field}.")
        elif value not in raw[field].unique():
            raise HTTPException(422, detail=f"Choose a valid value for {field}.")
    row = defaults | patient.features
    frame = pd.DataFrame([row], columns=raw.columns)
    probabilities = {name: round(float(model.predict_proba(frame)[0, 1]), 4) for name, model in models.items()}
    # Exact additive CAD evidence relative to the displayed reference profile.
    prep = models["CAD"].named_steps["preprocess"]
    classifier = models["CAD"].named_steps["classifier"]
    reference = pd.DataFrame([defaults], columns=raw.columns)
    delta = np.asarray(prep.transform(frame) - prep.transform(reference)).ravel() * classifier.coef_[0]
    contributions = {c: 0.0 for c in raw.columns}
    for encoded, contribution in zip(prep.get_feature_names_out(), delta):
        name = encoded.split("__", 1)[1]
        field = name if encoded.startswith("num__") else next(c for c in sorted(raw.columns, key=len, reverse=True) if name.startswith(c + "_"))
        contributions[field] += float(contribution)
    explanation = sorted([{"feature": c, "value": "Female" if row[c] == "Fmale" else row[c], "contribution": round(contributions[c], 6)} for c in patient.features], key=lambda x: abs(x["contribution"]), reverse=True)
    return {"probabilities": probabilities, "cad_status": "Positive screen" if probabilities["CAD"] >= .5 else "Negative screen", "explanation": explanation, "reference_probability": float(models["CAD"].predict_proba(reference)[0, 1]), "explanation_method": "Exact additive change in CAD log-odds relative to the cohort reference profile; association, not causation.", "imputed_from_reference": sorted(set(defaults) - set(patient.features)), "disclaimer": "Decision support only; not a diagnosis."}
