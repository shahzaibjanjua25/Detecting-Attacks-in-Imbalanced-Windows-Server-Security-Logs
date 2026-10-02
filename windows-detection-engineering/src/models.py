"""
Train and evaluate multiple detection models on Windows Security events.

Models:
1. Majority Baseline (DummyClassifier) — proves accuracy is misleading
2. Naive Bayes (GaussianNB) — baseline probabilistic classifier
3. XGBoost (with class weighting) — industry-standard strong baseline
4. Isolation Forest — unsupervised anomaly detection

Metrics:
- Accuracy (reported but misleading)
- Precision, Recall, F1
- PR-AUC (primary metric for imbalanced data)
- ROC-AUC
- Confusion matrix components

Outputs:
- results/model_comparison.csv
- results/feature_importance.csv
- results/pr_curves.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)
from xgboost import XGBClassifier

from features import prepare_datasets

SEED = 42


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_model(name, y_true, y_pred, y_proba=None):
    """Compute all relevant metrics for imbalanced classification."""
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba) if y_proba is not None else np.nan,
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan,
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel()
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Make sure results directory exists
    os.makedirs("results", exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test = prepare_datasets()

    print(f"Training set: {X_train.shape}")
    print(f"Test set:     {X_test.shape}")
    print(f"Attack rate:  {y_train.mean():.4f}\n")

    results = []
    probas = {}  # store probability scores for PR curve plot

    # -----------------------------------------------------------------------
    # Model 0: Majority Baseline
    # -----------------------------------------------------------------------
    print("Training Majority Baseline...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    # DummyClassifier with most_frequent has no meaningful proba; use constant 0
    y_proba = np.zeros(len(y_test))
    results.append(evaluate_model("Majority Baseline", y_test, y_pred, y_proba))
    probas["Majority Baseline"] = y_proba

    # -----------------------------------------------------------------------
    # Model 1: Naive Bayes
    # -----------------------------------------------------------------------
    print("Training Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    y_proba = nb.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("Naive Bayes", y_test, y_pred, y_proba))
    probas["Naive Bayes"] = y_proba

    # -----------------------------------------------------------------------
    # Model 2: XGBoost with class weighting
    # -----------------------------------------------------------------------
    print("Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("XGBoost", y_test, y_pred, y_proba))
    probas["XGBoost"] = y_proba

    # -----------------------------------------------------------------------
    # Model 3: Isolation Forest
    # -----------------------------------------------------------------------
    print("Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.03,  # approximate expected attack ratio
        random_state=SEED,
    )
    iso.fit(X_train)
    # IsolationForest: -1 = anomaly, 1 = normal
    y_pred = (iso.predict(X_test) == -1).astype(int)
    # Anomaly score: lower = more anomalous. We negate so higher = more anomalous.
    raw_scores = -iso.score_samples(X_test)
    # Normalize to [0, 1] so it can be treated as a probability-like score
    y_proba = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    results.append(evaluate_model("Isolation Forest", y_test, y_pred, y_proba))
    probas["Isolation Forest"] = y_proba

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    df = pd.DataFrame(results)
    print("\n" + "=" * 100)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 100)
    print(df.to_string(index=False))
    print()

    df.to_csv("results/model_comparison.csv", index=False)

    # -----------------------------------------------------------------------
    # Feature importance (XGBoost)
    # -----------------------------------------------------------------------
    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": xgb.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("XGBoost Feature Importance:")
    print(importance.to_string(index=False))
    print()
    importance.to_csv("results/feature_importance.csv", index=False)

    # -----------------------------------------------------------------------
    # Precision-Recall curves
    # -----------------------------------------------------------------------
    print("Generating PR curves...")
    plt.figure(figsize=(9, 7))

    # Plot PR curve for each model that has meaningful proba scores
    for name, y_proba in probas.items():
        if name == "Majority Baseline":
            continue  # constant proba, no curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        plt.plot(recall, precision, linewidth=2, label=f"{name} (PR-AUC = {ap:.3f})")

    # Horizontal line: baseline = attack prevalence
    baseline = y_test.mean()
    plt.axhline(
        y=baseline,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Random baseline (prevalence = {baseline:.3f})",
    )

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves — Windows Detection Engineering", fontsize=13)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig("results/pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved results/pr_curves.png")

    # -----------------------------------------------------------------------
    # Bar chart: F1 vs PR-AUC
    # -----------------------------------------------------------------------
    print("Generating model comparison bar chart...")
    fig, ax = plt.subplots(figsize=(10, 6))

    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.2

    metrics_to_plot = ["precision", "recall", "f1", "pr_auc"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        values = df[metric].fillna(0).tolist()
        ax.bar(x + i * width, values, width, label=metric.upper(), color=color)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison — Windows Detection Engineering", fontsize=13)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved results/model_comparison.png")

    # -----------------------------------------------------------------------
    # Final summary print
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    best_f1 = df.loc[df["f1"].idxmax()]
    best_prauc = df.loc[df["pr_auc"].idxmax()]
    print(f"Best F1:     {best_f1['model']} ({best_f1['f1']:.4f})")
    print(f"Best PR-AUC: {best_prauc['model']} ({best_prauc['pr_auc']:.4f})")
    print(f"\nAll results saved to results/")


if __name__ == "__main__":
    main()