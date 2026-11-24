"""
Step 3: Use SHAP to see WHY customers churn, and flag high-risk customers.
Run:  python explain.py   (after train.py)
"""
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load("outputs/xgb_model.pkl")
X_test = pd.read_csv("outputs/X_test.csv")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global view: which features matter most overall
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("outputs/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# Top drivers in plain numbers
importance = pd.Series(abs(shap_values).mean(axis=0), index=X_test.columns)
print("Top 8 churn drivers (mean |SHAP|):")
print(importance.sort_values(ascending=False).head(8).round(3))

# Local view: why is ONE customer risky?
probs = model.predict_proba(X_test)[:, 1]
result = X_test.copy()
result["churn_probability"] = probs
top = result.sort_values("churn_probability", ascending=False).head(10)
top.to_csv("outputs/high_risk_customers.csv", index=False)
print("\nSaved top 10 high-risk customers to outputs/high_risk_customers.csv")

idx = top.index[0]
shap.force_plot(explainer.expected_value, shap_values[idx],
                X_test.iloc[idx],
                matplotlib=True, show=False)
plt.savefig("outputs/shap_single_customer.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved SHAP plots in outputs/")
