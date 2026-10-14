# worker.py
import redis
import json
import time
import os
import joblib
import pandas as pd
import shap
from datetime import datetime, timezone
from database import SessionLocal, ForensicAudit, BatchJob

# 1. Load Model Pipeline
artifact = joblib.load("fraud_engine.joblib")
pipeline = artifact["pipeline"]
FRAUD_THRESHOLD = artifact["decision_threshold"]

preprocessor = pipeline.named_steps['preprocessor']
classifier = pipeline.named_steps['classifier']

cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
cat_cols = cat_encoder.get_feature_names_out(['category'])
feature_names = ['amount', 'velocity_15m', 'foreign_ip', 'device_trust'] + list(cat_cols)

explainer = shap.TreeExplainer(classifier)

# 2. Redis Stream Config
r = redis.Redis(host="localhost", port=6379, decode_responses=True)
AUDIT_STREAM = "fraud_audit_stream"
BATCH_STREAM = "batch_job_stream"

GROUP_NAME = "ml_workers"
CONSUMER_NAME = "worker_1"

for stream in [AUDIT_STREAM, BATCH_STREAM]:
    try:
        r.xgroup_create(stream, GROUP_NAME, id="0", mkstream=True)
    except redis.exceptions.ResponseError:
        pass

print(f"[*] Worker '{CONSUMER_NAME}' listening on streams [{AUDIT_STREAM}, {BATCH_STREAM}]...")

def process_batch_job(job_id: str, file_path: str):
    """Processes large CSVs in chunks to maintain bounded RAM usage."""
    db = SessionLocal()
    job = db.query(BatchJob).filter(BatchJob.job_id == job_id).first()
    
    try:
        job.status = "PROCESSING"
        db.commit()

        # Count total rows
        total_rows = sum(1 for _ in open(file_path)) - 1
        job.total_records = total_rows
        db.commit()

        output_file = f"results/scored_{job_id}.csv"
        chunk_size = 1000
        processed = 0
        fraud_count = 0
        is_first_chunk = True

        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Model scoring
            probs = pipeline.predict_proba(chunk)[:, 1]
            verdicts = [
                "REJECT" if p >= FRAUD_THRESHOLD 
                else "CHALLENGE" if p >= (FRAUD_THRESHOLD * 0.6) 
                else "APPROVE" 
                for p in probs
            ]

            chunk['risk_probability'] = probs.round(4)
            chunk['verdict'] = verdicts
            fraud_count += sum(1 for v in verdicts if v == "REJECT")
            processed += len(chunk)

            # Append chunk to output file
            chunk.to_csv(output_file, mode='w' if is_first_chunk else 'a', index=False, header=is_first_chunk)
            is_first_chunk = False

            # Update database progress for polling
            job.processed_records = processed
            job.fraud_detected = fraud_count
            db.commit()

        job.status = "COMPLETED"
        job.completed_at = datetime.now(timezone.utc)
        job.output_filepath = output_file
        db.commit()
        print(f"[✓] Batch Job {job_id} Completed: {processed} rows scored ({fraud_count} frauds).")

    except Exception as e:
        job.status = "FAILED"
        job.error_message = str(e)
        db.commit()
        print(f"[!] Batch Job {job_id} Failed: {e}")
    finally:
        db.close()

while True:
    try:
        # Read from both streams
        streams = {AUDIT_STREAM: ">", BATCH_STREAM: ">"}
        entries = r.xreadgroup(GROUP_NAME, CONSUMER_NAME, streams, count=1, block=2000)

        if not entries:
            continue

        for stream_name, messages in entries:
            for message_id, data in messages:
                if stream_name == AUDIT_STREAM:
                    # Forensic single audit
                    tx_id = data["transaction_id"]
                    verdict = data["verdict"]
                    risk_prob = float(data["risk_probability"])
                    raw_payload = json.loads(data["payload"])

                    input_df = pd.DataFrame([raw_payload])
                    X_trans = preprocessor.transform(input_df)
                    shap_vals = explainer.shap_values(X_trans)[0]

                    drivers = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:3]
                    top_drivers = [{col: round(float(val), 4)} for col, val in drivers]

                    db = SessionLocal()
                    audit_entry = ForensicAudit(
                        transaction_id=tx_id,
                        verdict=verdict,
                        risk_probability=risk_prob,
                        top_risk_drivers=top_drivers,
                        raw_payload=raw_payload
                    )
                    db.add(audit_entry)
                    db.commit()
                    db.close()

                    r.xack(AUDIT_STREAM, GROUP_NAME, message_id)
                    print(f"[+] Audited Single Txn: {tx_id}")

                elif stream_name == BATCH_STREAM:
                    # Batch job execution
                    job_id = data["job_id"]
                    file_path = data["file_path"]
                    print(f"[*] Starting Batch Job: {job_id}")
                    process_batch_job(job_id, file_path)
                    r.xack(BATCH_STREAM, GROUP_NAME, message_id)

    except Exception as e:
        print(f"[!] Worker Loop Error: {e}")
        time.sleep(1)