# Customer Churn Prediction

End-to-end churn prediction project: feature engineering, model comparison
(Logistic Regression, Random Forest, tuned XGBoost) and SHAP interpretability
to find *why* customers leave and *who* to target for retention.

**Tech:** Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib

## Project structure
```
churn_project/
├── generate_data.py     # makes data/churn.csv (or use the real Telco CSV)
├── train.py             # cleaning, features, models, tuning, evaluation
├── explain.py           # SHAP explanations + high-risk customer list
├── requirements.txt
├── interview_prep.md
├── data/churn.csv
└── outputs/             # plots, metrics, saved model
```

## How to run (VS Code terminal)
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

python generate_data.py        # skip if you use your own data/churn.csv
python train.py
python explain.py
```

## Pipeline
1. **Data**: 5,000 customers - contract, tenure, internet service, charges, support calls, etc.
2. **Features**: avg monthly spend, tenure groups, high-support-calls flag, risky combo (month-to-month + no tech support).
3. **Models**: Logistic Regression and Random Forest as baselines; XGBoost tuned with GridSearchCV (ROC-AUC, 3-fold CV) and `scale_pos_weight` for class imbalance.
4. **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC (churn is imbalanced, so accuracy alone is not enough).
5. **SHAP**: global drivers (summary/bar plots) and per-customer explanation.

## Key findings
- Short contracts (month-to-month) and low tenure are the strongest churn drivers.
- Fiber-optic customers, frequent support callers and no-tech-support customers are higher risk.
- Retention ideas: offer discounts for moving to 1-2 year contracts, proactive outreach after repeated support calls, push tech-support bundles.

## Using the real dataset (recommended)
Download "Telco Customer Churn" from Kaggle, save as `data/churn.csv`, and rename
columns to match (`tenure`->`tenure_months`, `Contract`->`contract`, etc.) or adjust
the names in `train.py`. Quote your final metrics from the real run.

## Note
The bundled dataset is synthetic, so the metrics in `outputs/` are for demonstration.
