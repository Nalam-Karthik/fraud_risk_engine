# app.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import redis
import json
import time
import uuid
import os
from typing import Optional
from database import SessionLocal, BatchJob

app = FastAPI(title="High-Throughput Fraud & Risk Engine")

# Create storage directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. Load ML Pipeline & Threshold
artifact = joblib.load("fraud_engine.joblib")
pipeline = artifact["pipeline"]
FRAUD_THRESHOLD = artifact["decision_threshold"]

# 2. Redis Connection
r = redis.Redis(host="localhost", port=6379, decode_responses=True)
STREAM_KEY = "fraud_audit_stream"
BATCH_STREAM_KEY = "batch_job_stream"

class TransactionPayload(BaseModel):
    transaction_id: str
    amount: float = Field(..., gt=0.0)
    velocity_15m: int = Field(..., ge=0)
    foreign_ip: int = Field(..., ge=0, le=1)
    device_trust: Optional[float] = Field(None, ge=0.0, le=1.0)
    category: str

# --- Real-Time Single Transaction Evaluation ---
@app.post("/v1/transactions/evaluate")
def evaluate_transaction(txn: TransactionPayload):
    t_start = time.perf_counter()
    input_dict = txn.model_dump()

    tx_df = pd.DataFrame([{
        'amount': input_dict['amount'],
        'velocity_15m': input_dict['velocity_15m'],
        'foreign_ip': input_dict['foreign_ip'],
        'device_trust': input_dict['device_trust'],
        'category': input_dict['category']
    }])

    risk_prob = float(pipeline.predict_proba(tx_df)[0][1])

    if risk_prob >= FRAUD_THRESHOLD:
        verdict = "REJECT"
    elif risk_prob >= (FRAUD_THRESHOLD * 0.6):
        verdict = "CHALLENGE"
    else:
        verdict = "APPROVE"

    latency_ms = (time.perf_counter() - t_start) * 1000

    try:
        r.xadd(STREAM_KEY, {
            "transaction_id": txn.transaction_id,
            "risk_probability": str(round(risk_prob, 4)),
            "verdict": verdict,
            "payload": json.dumps(input_dict)
        })
    except Exception as e:
        print(f"[BROKER ERROR] Failed to push event: {e}")

    return {
        "transaction_id": txn.transaction_id,
        "verdict": verdict,
        "risk_probability": round(risk_prob, 4),
        "decision_latency_ms": round(latency_ms, 2)
    }

# --- Batch CSV Async Upload Endpoint ---
@app.post("/v1/transactions/batch-evaluate", status_code=202)
async def upload_batch_transactions(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    saved_path = f"uploads/{job_id}.csv"

    # Stream file to disk in chunks to avoid blowing up memory
    with open(saved_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            buffer.write(chunk)

    # Register Job in Database
    db = SessionLocal()
    new_job = BatchJob(job_id=job_id, status="PENDING")
    db.add(new_job)
    db.commit()
    db.close()

    # Emit Batch Task to Redis Stream
    r.xadd(BATCH_STREAM_KEY, {
        "job_id": job_id,
        "file_path": saved_path
    })

    return {
        "job_id": job_id,
        "status": "PENDING",
        "message": "File received. Processing asynchronously.",
        "poll_status_url": f"/v1/transactions/jobs/{job_id}"
    }

# --- Batch Job Status Polling Endpoint ---
@app.get("/v1/transactions/jobs/{job_id}")
def check_job_status(job_id: str):
    db = SessionLocal()
    job = db.query(BatchJob).filter(BatchJob.job_id == job_id).first()
    db.close()

    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    progress_percent = 0.0
    if job.total_records > 0:
        progress_percent = round((job.processed_records / job.total_records) * 100, 2)

    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": f"{progress_percent}%",
        "processed_records": job.processed_records,
        "total_records": job.total_records,
        "fraud_detected": job.fraud_detected,
        "download_url": f"/v1/transactions/jobs/{job_id}/download" if job.status == "COMPLETED" else None,
        "error_message": job.error_message
    }

# --- Download Scored CSV Endpoint ---
@app.get("/v1/transactions/jobs/{job_id}/download")
def download_batch_results(job_id: str):
    db = SessionLocal()
    job = db.query(BatchJob).filter(BatchJob.job_id == job_id).first()
    db.close()

    if not job or job.status != "COMPLETED":
        raise HTTPException(status_code=400, detail="Job is not completed yet.")

    if not os.path.exists(job.output_filepath):
        raise HTTPException(status_code=404, detail="File missing from storage.")

    return FileResponse(
        path=job.output_filepath,
        filename=f"scored_{job_id}.csv",
        media_type="text/csv"
    )