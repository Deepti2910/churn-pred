# Interview Prep - Customer Churn Prediction

## 30-second pitch
"I built an end-to-end churn prediction pipeline. I cleaned the customer data,
engineered features like average monthly spend and tenure groups, compared
Logistic Regression, Random Forest and a tuned XGBoost, and used SHAP to explain
the drivers. The goal was not only to predict who will leave, but to tell the
business why, so retention offers can be targeted."

## The story, step by step
1. Problem: losing customers costs more than keeping them, so flag risky ones early.
2. Data: ~5k customers, ~39% churn (imbalanced).
3. Cleaning: fix blank charges, encode target, one-hot encode categoricals.
4. Features: avg spend, tenure bucket, high support calls, month-to-month + no support combo.
5. Models: simple baseline first, then tree models.
6. Tuning: GridSearchCV on max_depth, learning_rate, n_estimators, scored on ROC-AUC.
7. Evaluation: recall + precision + ROC-AUC, not accuracy alone.
8. Explainability: SHAP for global drivers and one-customer explanations.
9. Action: retention suggestions from the drivers.

## Likely questions and answers

**Why XGBoost?**
It handles non-linear relationships and feature interactions, works well on tabular data, has built-in regularization, and supports class weighting. It was also the best on recall in my comparison.

**Why not just accuracy?**
Churners are the minority. A model predicting "nobody churns" can still look accurate. I used recall (catch churners), precision (avoid wasting offers), F1 and ROC-AUC.

**Precision vs recall - which matters more here?**
Usually recall, because missing a churner loses revenue. But if retention offers are expensive, precision matters too. It is a business trade-off; I can adjust the probability threshold.

**How did you handle class imbalance?**
Stratified train-test split and `scale_pos_weight` in XGBoost. Other options: SMOTE, class weights, threshold tuning.

**What is ROC-AUC?**
The probability that the model ranks a random churner above a random non-churner. 0.5 is random, 1.0 is perfect.

**What is SHAP?**
It assigns each feature a contribution to each prediction, based on Shapley values from game theory. Positive SHAP pushes toward churn, negative away. It gives both global importance and per-customer explanations.

**SHAP vs feature_importances_?**
Built-in importance only shows how much a feature was used in splits, with no direction and no per-customer view. SHAP gives direction, consistency and local explanations.

**Did you check for overfitting?**
I used a held-out test set and 3-fold cross-validation during tuning, and kept trees shallow (max_depth 3-5).

**What is data leakage and did you avoid it?**
Using information in training that would not be available at prediction time. I fit the scaler only on training data and split before scaling.

**Why scale for Logistic Regression but not trees?**
Logistic Regression is sensitive to feature scale; tree models split on thresholds so scale does not matter.

**Why one-hot encoding with drop_first?**
Categories have no order, so one-hot avoids a fake ranking. drop_first avoids the dummy variable trap (perfect multicollinearity).

**How would you deploy it?**
Save the model with joblib, wrap it in a FastAPI/Flask endpoint, score customers weekly, and send the high-risk list to the retention team. Monitor for drift and retrain periodically.

**What would you improve?**
Real customer data with more behavioral signals (usage trends, complaints), time-based validation, probability calibration, and a cost-based threshold.

**Which features were most important?**
Contract type and tenure were the top drivers, then fiber internet, the risky combo feature, and support calls.

## Be honest about
- The bundled data is synthetic. If you run it on the real Telco dataset, quote those metrics.
- Know every line of the code: be ready to explain `train_test_split(stratify=...)`, `get_dummies`, and `GridSearchCV`.

## Quick concept refresher
- Precision = TP / (TP + FP); Recall = TP / (TP + FN); F1 = harmonic mean.
- Bagging (Random Forest) = parallel trees, reduces variance. Boosting (XGBoost) = sequential trees, reduces bias.
- Overfitting fix: shallower trees, regularization, early stopping, more data, CV.
