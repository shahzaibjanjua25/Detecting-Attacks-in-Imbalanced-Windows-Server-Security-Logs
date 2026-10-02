"""
Train and evaluate multiple detection models.

Models:
1. Naive Bayes (baseline, interpretable)
2. XGBoost (industry-standard, strong baseline)
3. Isolation Forest (unsupervised anomaly detection)
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier
from features import prepare_datasets

SEED = 42


def evaluate_model(name, y_true, y_pred, y_proba=None):
    """Compute all relevant metrics for imbalanced classification."""
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba) if y_proba is not None else None,
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else None,
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel()
    return metrics


def main():
    X_train, X_test, y_train, y_test = prepare_datasets()

    print(f"Training set: {X_train.shape}")
    print(f"Test set:     {X_test.shape}")
    print(f"Attack rate:  {y_train.mean():.4f}\n")

    results = []

    # ---- Model 1: Naive Bayes ----
    print("Training Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    y_proba = nb.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("Naive Bayes", y_test, y_pred, y_proba))

    # ---- Model 2: XGBoost with class weighting ----
    print("Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("XGBoost", y_test, y_pred, y_proba))

    # ---- Model 3: Isolation Forest ----
    print("Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.02,  # matches expected attack ratio
        random_state=SEED,
    )
    iso.fit(X_train)
    # IsolationForest: -1 = anomaly, 1 = normal
    y_pred = (iso.predict(X_test) == -1).astype(int)
    # Anomaly score: lower = more anomalous
    y_proba = -iso.score_samples(X_test)
    # Normalize to 0-1
    y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    results.append(evaluate_model("Isolation Forest", y_test, y_pred, y_proba))

    # ---- Summary ----
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))
    print()

    df.to_csv("results/model_comparison.csv", index=False)

    # Feature importance for XGBoost
    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": xgb.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nXGBoost Feature Importance:")
    print(importance.to_string(index=False))
    importance.to_csv("results/feature_importance.csv", index=False)


if __name__ == "__main__":
    main()