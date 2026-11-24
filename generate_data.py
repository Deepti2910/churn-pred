"""
Step 1: Create a telecom-style customer dataset.

If you have the real Kaggle "Telco Customer Churn" CSV, just put it in
data/churn.csv (with a 'Churn' column of Yes/No) and skip this file.
This script makes a realistic fake dataset so the project runs anywhere.
"""
import os
import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

df = pd.DataFrame({
    "customer_id": [f"C{i:05d}" for i in range(n)],
    "gender": np.random.choice(["Male", "Female"], n),
    "senior_citizen": np.random.choice([0, 1], n, p=[0.84, 0.16]),
    "tenure_months": np.random.randint(1, 73, n),
    "contract": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]),
    "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20]),
    "tech_support": np.random.choice(["Yes", "No"], n, p=[0.35, 0.65]),
    "payment_method": np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
    "monthly_charges": np.round(np.random.uniform(20, 115, n), 2),
    "support_calls": np.random.poisson(1.5, n),
})
df["total_charges"] = np.round(df["monthly_charges"] * df["tenure_months"], 2)

# Churn probability depends on a few realistic drivers
score = -1.2
score += (df["contract"] == "Month-to-month") * 1.4
score -= (df["contract"] == "Two year") * 1.0
score += (df["internet_service"] == "Fiber optic") * 0.6
score += (df["tech_support"] == "No") * 0.5
score += (df["payment_method"] == "Electronic check") * 0.5
score += df["support_calls"] * 0.25
score -= df["tenure_months"] * 0.03
score += (df["monthly_charges"] - 65) * 0.012
prob = 1 / (1 + np.exp(-score))
df["Churn"] = np.where(np.random.rand(n) < prob, "Yes", "No")

os.makedirs("data", exist_ok=True)
df.to_csv("data/churn.csv", index=False)
print("Saved data/churn.csv", df.shape)
print(df["Churn"].value_counts(normalize=True).round(3))
