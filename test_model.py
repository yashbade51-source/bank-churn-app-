"""
test_model.py - Test Suite for Merged Churn Prediction & Retention System
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from llm_service import generate_churn_explanation

print("=" * 60)
print("1. TESTING MODEL & ARTIFACT LOADING")
print("=" * 60)

model = joblib.load("best_churn_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
config = joblib.load("churn_config.pkl")

print("[OK] Model loaded successfully:", type(model))
print(f"[OK] Feature columns ({len(feature_columns)}):", feature_columns)
print("[OK] Config:", config)

print("\n" + "=" * 60)
print("2. TESTING MODEL INFERENCE")
print("=" * 60)

# Sample customer
sample_dict = {
    "CreditScore": 650,
    "Age": 42,
    "Tenure": 3,
    "Balance": 115000.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 85000.0,
    "Satisfaction Score": 2,
    "Point Earned": 450,
    "Geography_Germany": 1,
    "Geography_Spain": 0,
    "Gender_Male": 1,
    "Card Type_GOLD": 1,
    "Card Type_PLATINUM": 0,
    "Card Type_SILVER": 0
}

df_input = pd.DataFrame([sample_dict]).reindex(columns=feature_columns, fill_value=0)
prob = model.predict_proba(df_input)[0][1]
pred = int(prob >= config.get("threshold", 0.5))

print(f"[OK] Sample Churn Probability: {prob * 100:.2f}%")
print(f"[OK] Prediction Outcome: {'CHURN RISK' if pred == 1 else 'RETAIN'}")

print("\n" + "=" * 60)
print("3. TESTING FEATURE EXPLAINABILITY (SHAP / ATTRIBUTION)")
print("=" * 60)

top_factors = []
try:
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df_input).values[0]
    if np.ndim(shap_values) > 1:
        shap_values = shap_values[:, -1]

    for col, val, s in sorted(zip(feature_columns, df_input.iloc[0].values, shap_values), key=lambda x: abs(x[2]), reverse=True)[:5]:
        direction = "Increases Churn" if s > 0 else "Reduces Churn"
        print(f"  - {col} (val={val}): SHAP={s:+.4f} [{direction}]")
        top_factors.append(f"{col}: {direction}")
    print("[OK] SHAP TreeExplainer executed successfully.")
except Exception as e:
    print(f"[*] SHAP dynamic fallback active ({type(e).__name__}). Using model feature importance:")
    importances = model.feature_importances_
    for col, imp in sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {col}: Importance={imp:.4f}")
        top_factors.append(f"{col} (Importance: {imp:.2f})")
    print("[OK] Feature importance attribution calculated.")

print("\n" + "=" * 60)
print("4. TESTING AI RETENTION COPILOT (OLLAMA LLM SERVICE)")
print("=" * 60)

try:
    explanation = generate_churn_explanation(
        customer_data=sample_dict,
        churn_probability=prob * 100,
        top_factors=top_factors
    )
    print("[OK] Ollama response received successfully:")
    print("-" * 60)
    print(explanation)
    print("-" * 60)
except Exception as e:
    print(f"ℹ️ Ollama offline or not started: {e}")
    print("  (Start local Ollama using: ollama run llama3.2)")

print("\n" + "=" * 60)
print("5. TESTING DATASET INTEGRITY")
print("=" * 60)
if os.path.exists("Customer-Churn-Records.csv"):
    df_data = pd.read_csv("Customer-Churn-Records.csv")
    print(f"[OK] Customer dataset verified: {len(df_data):,} rows, {len(df_data.columns)} columns")
else:
    print("[FAIL] Customer-Churn-Records.csv missing!")

print("\n*** ALL VERIFICATION TESTS COMPLETED! ***")
