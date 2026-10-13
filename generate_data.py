# generate_data.py
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 5000

# 1. Feature generation
amounts = np.random.exponential(scale=75, size=n_samples) + 5.0
velocity_15m = np.random.poisson(lam=1.5, size=n_samples)
foreign_ip = np.random.binomial(n=1, p=0.08, size=n_samples)
device_trust = np.random.beta(a=8, b=2, size=n_samples) # Most devices trusted (0.8 - 1.0)
categories = np.random.choice(
    ['groceries', 'electronics', 'crypto_exchange', 'luxury', 'dining'], 
    size=n_samples, 
    p=[0.4, 0.2, 0.05, 0.1, 0.25]
)

# 2. Add realistic missing values (e.g. device_trust failed to resolve)
mask_missing_trust = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
device_trust[mask_missing_trust] = np.nan

# 3. Ground Truth Fraud Logic (Severe Imbalance: ~2% Fraud)
fraud_probability = (
    (amounts > 400).astype(float) * 0.35 +
    (velocity_15m > 4).astype(float) * 0.40 +
    (foreign_ip == 1).astype(float) * 0.25 +
    (np.nan_to_num(device_trust, nan=0.2) < 0.4).astype(float) * 0.45 +
    (categories == 'crypto_exchange').astype(float) * 0.30
)

# Clip between 0 and 1
fraud_probability = np.clip(fraud_probability, 0.0, 0.95)
is_fraud = (np.random.binomial(n=1, p=fraud_probability) == 1).astype(int)

df = pd.DataFrame({
    'amount': np.round(amounts, 2),
    'velocity_15m': velocity_15m,
    'foreign_ip': foreign_ip,
    'device_trust': np.round(device_trust, 3),
    'category': categories,
    'is_fraud': is_fraud
})

print(f"Total Transactions: {len(df)}")
print(f"Fraud Cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean() * 100:.2f}%)")

df.to_csv("transactions.csv", index=False)
print("Saved to 'transactions.csv'!")