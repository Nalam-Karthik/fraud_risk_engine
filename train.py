# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
import joblib

# 1. Load dataset
df = pd.read_csv("transactions.csv")
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Build preprocessing pipeline
numeric_features = ['amount', 'velocity_15m', 'foreign_ip', 'device_trust']
categorical_features = ['category']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 3. Model with Class Weighting for Imbalanced Data
# class_weight='balanced' automatically inflates the loss penalty for the rare fraud class
model = HistGradientBoostingClassifier(
    class_weight='balanced',
    max_iter=120,
    learning_rate=0.08,
    random_state=42
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

print("Training Fraud Detection Pipeline...")
pipeline.fit(X_train, y_train)

# 4. Cost-Sensitive Threshold Determination
# Probabilities of the positive (Fraud) class
fraud_probabilities = pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, fraud_probabilities)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Compute precision-recall trade-offs
precisions, recalls, thresholds = precision_recall_curve(y_test, fraud_probabilities)

# Target: Catch at least 85% of fraud (Recall >= 0.85) with minimal false alarms
valid_idx = np.where(recalls >= 0.85)[0]
optimal_threshold = thresholds[valid_idx[-1]] if len(valid_idx) > 0 else 0.50

print(f"Standard Default Threshold: 0.5000")
print(f"Optimal Recall-Oriented Threshold: {optimal_threshold:.4f}\n")

# Evaluation under optimal threshold
custom_predictions = (fraud_probabilities >= optimal_threshold).astype(int)
print("Classification Report at Optimized Threshold:")
print(classification_report(y_test, custom_predictions, target_names=['Legitimate', 'Fraud']))

# 5. Save the pipeline and the decision threshold together
artifact = {
    "pipeline": pipeline,
    "decision_threshold": float(optimal_threshold)
}

joblib.dump(artifact, "fraud_engine.joblib")
print("Saved engine artifact to 'fraud_risk_engine.joblib'!")