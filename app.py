# app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import json
import time
import shap
from typing import Optional
from datetime import datetime, timezone

app = FastAPI(title="High-Throughput Fraud & Risk Engine")

# 1. Load artifacts into memory on startup
artifact = joblib.load("fraud_engine.joblib")
pipeline = artifact["pipeline"]
FRAUD_THRESHOLD = artifact["decision_threshold"]

# Prepare SHAP explainer for background worker
preprocessor = pipeline.named_steps['preprocessor']
classifier = pipeline.named_steps['classifier']

# Extract one-hot encoded feature names
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
cat_cols = cat_encoder.get_feature_names_out(['category'])
feature_names = ['amount', 'velocity_15m', 'foreign_ip', 'device_trust'] + list(cat_cols)

explainer = shap.TreeExplainer(classifier)

# 2. Request Data Contract (Strict Pydantic Validation)
class TransactionPayload(BaseModel):
    transaction_id: str
    amount: float = Field(..., gt=0.0, description="Amount in USD")
    velocity_15m: int = Field(..., ge=0, description="Count of txns in last 15 mins")
    foreign_ip: int = Field(..., ge=0, le=1, description="1 if IP is outside home country, else 0")
    device_trust: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score of user device")
    category: str = Field(..., description="Merchant category code")

# 3. Asynchronous Worker Routine (Runs out-of-band)
def audit_forensics_worker(tx_data: dict, risk_prob: float, verdict: str):
    """Heavy forensic calculation detached from HTTP request loop."""
    try:
        tx_df = pd.DataFrame([tx_data])
        X_trans = preprocessor.transform(tx_df)
        shap_vals = explainer.shap_values(X_trans)[0]

        # Top 3 drivers for explainability
        drivers = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:3]
        top_explanations = [f"{col}: {val:+.3f}" for col, val in drivers]

        audit_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "transaction_id": tx_data.get("transaction_id"),
            "risk_probability": round(risk_prob, 4),
            "verdict": verdict,
            "top_risk_drivers": top_explanations,
            "raw_payload": tx_data
        }

        # Write to append-only JSONL log (Forensics store)
        with open("audit_trail.jsonl", "a") as f:
            f.write(json.dumps(audit_record) + "\n")

    except Exception as e:
        print(f"[AUDIT WORKER ERROR] Failed to record forensics: {e}")

# 4. Synchronous High-Speed Evaluation Endpoint
@app.post("/v1/transactions/evaluate")
def evaluate_transaction(txn: TransactionPayload, background_tasks: BackgroundTasks):
    t_start = time.perf_counter()

    # Convert Pydantic model to DataFrame
    input_dict = txn.model_dump()
    tx_df = pd.DataFrame([{
        'amount': input_dict['amount'],
        'velocity_15m': input_dict['velocity_15m'],
        'foreign_ip': input_dict['foreign_ip'],
        'device_trust': input_dict['device_trust'],
        'category': input_dict['category']
    }])

    # Score model
    risk_prob = float(pipeline.predict_proba(tx_df)[0][1])

    # Three-Tier Risk Policy
    # High risk: immediate REJECT
    # Moderate risk: CHALLENGE (trigger 2FA / OTP)
    # Low risk: APPROVE
    if risk_prob >= FRAUD_THRESHOLD:
        verdict = "REJECT"
    elif risk_prob >= (FRAUD_THRESHOLD * 0.6):
        verdict = "CHALLENGE"
    else:
        verdict = "APPROVE"

    latency_ms = (time.perf_counter() - t_start) * 1000

    # Offload SHAP computation and disk write to background worker
    background_tasks.add_task(
        audit_forensics_worker,
        tx_data=input_dict,
        risk_prob=risk_prob,
        verdict=verdict
    )

    return {
        "transaction_id": txn.transaction_id,
        "verdict": verdict,
        "risk_probability": round(risk_prob, 4),
        "threshold_applied": round(FRAUD_THRESHOLD, 4),
        "decision_latency_ms": round(latency_ms, 2)
    }