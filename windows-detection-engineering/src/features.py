"""
Feature engineering with strict train/test separation.

CRITICAL: All statistical thresholds are derived from TRAINING data only.
This prevents target leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42


def load_data(path="data/windows_security_events.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def engineer_features(df, train_mask):
    """
    Engineer features for the full dataset, using thresholds
    derived ONLY from the training subset.

    Args:
        df: full dataframe
        train_mask: boolean mask indicating training rows

    Returns:
        feature dataframe
    """
    df = df.copy()

    # ---- Step 1: Temporal features (no leakage, purely timestamp-based) ----
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_business_hours"] = (
        (df["hour"] >= 9) & (df["hour"] <= 17) & (df["is_weekend"] == 0)
    ).astype(int)

    # ---- Step 2: Event-based features (no leakage, purely event_id-based) ----
    df["is_auth_event"] = df["event_id"].isin([4624, 4625, 4634, 4648]).astype(int)
    df["is_privileged_event"] = df["event_id"].isin([4672, 4673, 4720, 4732]).astype(int)

    # ---- Step 3: Text-based features (no leakage, purely message-based) ----
    df["message_length"] = df["message"].str.len()
    df["word_count"] = df["message"].str.split().str.len()

    # ---- Step 4: Derived features from TRAINING data ONLY ----
    # NOW we can use train_mask because hour/event_id columns exist
    train = df[train_mask]
    train_benign = train[train["is_malicious"] == 0]

    # Suspicious event IDs: those that are rare among BENIGN training events
    benign_event_counts = train_benign["event_id"].value_counts()
    # Event IDs appearing < 5% of the most common benign event are "rare"
    threshold = benign_event_counts.max() * 0.05
    rare_event_ids = set(benign_event_counts[benign_event_counts < threshold].index)

    df["is_rare_event"] = df["event_id"].isin(rare_event_ids).astype(int)

    # Unusual hour: derived from training benign events
    train_benign_hours = train_benign["hour"]
    hour_mean = train_benign_hours.mean()
    hour_std = train_benign_hours.std()

    # Handle case where std is 0 (shouldn't happen, but safe)
    if hour_std == 0:
        hour_std = 1.0

    df["is_unusual_hour"] = (
        (df["hour"] < hour_mean - 2 * hour_std)
        | (df["hour"] > hour_mean + 2 * hour_std)
    ).astype(int)

    # ---- Step 5: Select final feature columns ----
    feature_cols = [
        "hour",
        "day_of_week",
        "is_weekend",
        "is_business_hours",
        "event_id",
        "logon_type",
        "is_auth_event",
        "is_privileged_event",
        "message_length",
        "word_count",
        "is_rare_event",
        "is_unusual_hour",
    ]

    return df[feature_cols + ["is_malicious"]]


def prepare_datasets(test_size=0.2):
    """Full pipeline: load, split, engineer features."""
    df = load_data()

    # Split FIRST, before any feature engineering
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df["is_malicious"],
        random_state=SEED,
    )

    train_mask = df.index.isin(train_idx)

    # Engineer features using only training thresholds
    features = engineer_features(df, train_mask)

    X_train = features.loc[train_idx].drop(columns=["is_malicious"])
    y_train = features.loc[train_idx, "is_malicious"]
    X_test = features.loc[test_idx].drop(columns=["is_malicious"])
    y_test = features.loc[test_idx, "is_malicious"]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_datasets()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train attack rate: {y_train.mean():.4f}")
    print(f"Test attack rate:  {y_test.mean():.4f}")
    print(f"\nFeatures:\n{X_train.columns.tolist()}")
    print(f"\nFeature sample:\n{X_train.head().to_string()}")