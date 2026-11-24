"""
Step 2: Clean data, build features, train models, compare, save best.
Run:  python train.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

os.makedirs("outputs", exist_ok=True)

# ---------- 1. Load ----------
df = pd.read_csv("data/churn.csv")
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum().sum())

# Telco CSV sometimes has blank total charges - handle it safely
if "total_charges" in df.columns:
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["total_charges"] = df["total_charges"].fillna(df["total_charges"].median())

# ---------- 2. Target ----------
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df = df.drop(columns=["customer_id"])

# ---------- 3. Feature engineering ----------
# average spend per month of tenure
df["avg_monthly_spend"] = df["total_charges"] / df["tenure_months"]
# simple tenure buckets
df["tenure_group"] = pd.cut(df["tenure_months"], bins=[0, 12, 24, 48, 72],
                            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"])
# customers who call support a lot are often unhappy
df["high_support_calls"] = (df["support_calls"] >= 4).astype(int)
# risky combo: month-to-month + no tech support
df["risky_combo"] = ((df["contract"] == "Month-to-month") &
                     (df["tech_support"] == "No")).astype(int)

# ---------- 4. Encode categoricals ----------
df["tenure_group"] = df["tenure_group"].astype(str)
X = pd.get_dummies(df.drop(columns=["Churn"]), drop_first=True).astype(float)
y = df["Churn"]

# ---------- 5. Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Churn rate:", round(y.mean(), 3))

# ---------- 6. Baseline models ----------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

def evaluate(name, model, Xtr, Xte):
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    results[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }
    return model

evaluate("Logistic Regression", LogisticRegression(max_iter=1000), X_train_s, X_test_s)
evaluate("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42), X_train, X_test)

# ---------- 7. XGBoost with small hyperparameter search ----------
# scale_pos_weight handles class imbalance
spw = (y_train == 0).sum() / (y_train == 1).sum()
param_grid = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 200],
}
grid = GridSearchCV(
    XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", random_state=42),
    param_grid, scoring="roc_auc", cv=3, n_jobs=-1)
grid.fit(X_train, y_train)
print("Best XGBoost params:", grid.best_params_)
best_xgb = evaluate("XGBoost (tuned)", grid.best_estimator_, X_train, X_test)

# ---------- 8. Compare ----------
res_df = pd.DataFrame(results).T.round(3)
print("\nModel comparison:\n", res_df)
res_df.to_csv("outputs/model_comparison.csv")

pred = best_xgb.predict(X_test)
print("\nClassification report (XGBoost):\n", classification_report(y_test, pred))

# ---------- 9. Plots ----------
ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                       display_labels=["Stay", "Churn"]).plot(cmap="Blues")
plt.title("Confusion Matrix - XGBoost")
plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

RocCurveDisplay.from_estimator(best_xgb, X_test, y_test)
plt.title("ROC Curve - XGBoost")
plt.savefig("outputs/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- 10. Save for explain.py ----------
joblib.dump(best_xgb, "outputs/xgb_model.pkl")
X_test.to_csv("outputs/X_test.csv", index=False)
print("\nSaved model and plots in outputs/")
