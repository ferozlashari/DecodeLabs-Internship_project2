"""
server.py — FastAPI backend for the React frontend.

Endpoints:
    GET  /api/summary    -> dataset row/column info + OrderStatus counts
    GET  /api/sample     -> first N rows of the dataset (for a preview table)
    GET  /api/options    -> unique values for each dropdown field
    POST /api/train      -> train a KNN model with the given k / test_size,
                             returns accuracy, F1, confusion matrix
    POST /api/predict    -> predict OrderStatus for one order's details,
                             using the most recently trained model

Run with:
    uvicorn server:app --reload --port 8000
"""

from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from classifier.config import TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from classifier.data_loader import load_dataset
from classifier.preprocessing import split_features_and_target
from classifier.model import build_model_pipeline, split_train_test, evaluate_model

app = FastAPI(title="Project 2 — Order Status Classifier API")

# Allow the React dev server (Vite default port 5173) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup; the dataset doesn't change at runtime.
_df = load_dataset()

# Holds the most recently trained model, so /api/predict can use it.
_state = {"model": None}


class TrainRequest(BaseModel):
    k: int = 5
    test_size: float = 0.2


class PredictRequest(BaseModel):
    Quantity: float
    UnitPrice: float
    ItemsInCart: float
    TotalPrice: float
    Product: str
    PaymentMethod: str
    CouponCode: str
    ReferralSource: str


@app.get("/api/summary")
def get_summary():
    return {
        "rows": len(_df),
        "columns": list(_df.columns),
        "missing_values": int(_df.isna().sum().sum()),
        "order_status_counts": _df[TARGET_COLUMN].value_counts().to_dict(),
    }


@app.get("/api/sample")
def get_sample(rows: int = 10):
    sample = _df.head(rows)
    return sample.to_dict(orient="records")


@app.get("/api/options")
def get_options():
    return {
        "Product": sorted(_df["Product"].dropna().unique().tolist()),
        "PaymentMethod": sorted(_df["PaymentMethod"].dropna().unique().tolist()),
        "CouponCode": sorted(_df["CouponCode"].dropna().unique().tolist()),
        "ReferralSource": sorted(_df["ReferralSource"].dropna().unique().tolist()),
    }


@app.post("/api/train")
def train(req: TrainRequest):
    if not (1 <= req.k <= 50):
        raise HTTPException(400, "k must be between 1 and 50")
    if not (0.1 <= req.test_size <= 0.5):
        raise HTTPException(400, "test_size must be between 0.1 and 0.5")

    X, y = split_features_and_target(_df, TARGET_COLUMN)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=req.test_size)

    model = build_model_pipeline(n_neighbors=req.k)
    model.fit(X_train, y_train)
    results = evaluate_model(model, X_test, y_test)

    _state["model"] = model

    baseline = 1.0 / y.nunique()

    return {
        "accuracy": results["accuracy"],
        "f1_weighted": results["f1_weighted"],
        "labels": results["labels"],
        "confusion_matrix": results["confusion_matrix"].tolist(),
        "classification_report": results["classification_report"],
        "random_guess_baseline": baseline,
        "k": req.k,
        "test_size": req.test_size,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    model = _state["model"]
    if model is None:
        raise HTTPException(400, "Train a model first via POST /api/train")

    row = pd.DataFrame([req.dict()])
    prediction = model.predict(row)[0]

    response = {"prediction": prediction}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0]
        response["probabilities"] = {
            cls: float(p) for cls, p in zip(model.classes_, proba)
        }
    return response


@app.get("/api/health")
def health():
    return {"status": "ok"}
