# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import redis
import json
import time
from typing import Optional

app = FastAPI(title="Edge Risk Gateway")

# 1. Load ML Pipeline & Threshold
artifact = joblib.load("fraud_engine.joblib")
pipeline = artifact["pipeline"]
FRAUD_THRESHOLD = artifact["decision_threshold"]

# 2. Redis Connection (Message Broker)
r = redis.Redis(host="localhost", port=6379, decode_responses=True)
STREAM_KEY = "fraud_audit_stream"

class TransactionPayload(BaseModel):
    transaction_id: str
    amount: float = Field(..., gt=0.0)
    velocity_15m: int = Field(..., ge=0)
    foreign_ip: int = Field(..., ge=0, le=1)
    device_trust: Optional[float] = Field(None, ge=0.0, le=1.0)
    category: str

@app.post("/v1/transactions/evaluate")
def evaluate_transaction(txn: TransactionPayload):
    t_start = time.perf_counter()
    input_dict = txn.model_dump()

    # Create 1-row DataFrame for inference
    tx_df = pd.DataFrame([{
        'amount': input_dict['amount'],
        'velocity_15m': input_dict['velocity_15m'],
        'foreign_ip': input_dict['foreign_ip'],
        'device_trust': input_dict['device_trust'],
        'category': input_dict['category']
    }])

    # 3. Synchronous Scoring (<15ms)
    risk_prob = float(pipeline.predict_proba(tx_df)[0][1])

    if risk_prob >= FRAUD_THRESHOLD:
        verdict = "REJECT"
    elif risk_prob >= (FRAUD_THRESHOLD * 0.6):
        verdict = "CHALLENGE"
    else:
        verdict = "APPROVE"

    latency_ms = (time.perf_counter() - t_start) * 1000

    # 4. Publish Event to Redis Stream (Fire-and-Forget)
    event_payload = {
        "transaction_id": txn.transaction_id,
        "risk_probability": str(round(risk_prob, 4)),
        "verdict": verdict,
        "payload": json.dumps(input_dict)
    }
    
    try:
        r.xadd(STREAM_KEY, event_payload)
    except Exception as e:
        # Never fail a checkout response due to audit broker issues
        print(f"[BROKER ERROR] Failed to push to Redis: {e}")

    return {
        "transaction_id": txn.transaction_id,
        "verdict": verdict,
        "risk_probability": round(risk_prob, 4),
        "decision_latency_ms": round(latency_ms, 2)
    }