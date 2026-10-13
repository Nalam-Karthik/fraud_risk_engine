# worker.py
import redis
import json
import time
import joblib
import pandas as pd
import shap
from database import SessionLocal, ForensicAudit

# 1. Load Model & Explainer
artifact = joblib.load("fraud_engine.joblib")
pipeline = artifact["pipeline"]
preprocessor = pipeline.named_steps['preprocessor']
classifier = pipeline.named_steps['classifier']

cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
cat_cols = cat_encoder.get_feature_names_out(['category'])
feature_names = ['amount', 'velocity_15m', 'foreign_ip', 'device_trust'] + list(cat_cols)

explainer = shap.TreeExplainer(classifier)

# 2. Redis Stream Config
r = redis.Redis(host="localhost", port=6379, decode_responses=True)
STREAM_KEY = "fraud_audit_stream"
GROUP_NAME = "audit_workers"
CONSUMER_NAME = "worker_1"

# Initialize consumer group if not present
try:
    r.xgroup_create(STREAM_KEY, GROUP_NAME, id="0", mkstream=True)
except redis.exceptions.ResponseError:
    pass  # Group already exists

print(f"[*] Audit Worker '{CONSUMER_NAME}' listening on stream '{STREAM_KEY}'...")

while True:
    try:
        # Read new messages from the stream
        entries = r.xreadgroup(GROUP_NAME, CONSUMER_NAME, {STREAM_KEY: ">"}, count=5, block=2000)

        if not entries:
            continue

        db = SessionLocal()
        for stream_name, messages in entries:
            for message_id, data in messages:
                tx_id = data["transaction_id"]
                verdict = data["verdict"]
                risk_prob = float(data["risk_probability"])
                raw_payload = json.loads(data["payload"])

                # Compute SHAP forensics out-of-process
                input_df = pd.DataFrame([raw_payload])
                X_trans = preprocessor.transform(input_df)
                shap_vals = explainer.shap_values(X_trans)[0]

                drivers = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:3]
                top_drivers = [{col: round(float(val), 4)} for col, val in drivers]

                # Persist to relational database
                audit_entry = ForensicAudit(
                    transaction_id=tx_id,
                    verdict=verdict,
                    risk_probability=risk_prob,
                    top_risk_drivers=top_drivers,
                    raw_payload=raw_payload
                )
                db.add(audit_entry)
                db.commit()

                # Acknowledge message in Redis stream
                r.xack(STREAM_KEY, GROUP_NAME, message_id)
                print(f"[+] Processed & Audited: {tx_id} ({verdict})")

        db.close()

    except Exception as e:
        print(f"[!] Worker Loop Error: {e}")
        time.sleep(1)