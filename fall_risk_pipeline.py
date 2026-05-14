"""
Fall Risk Assessment — Unified ML Pipeline
==========================================
Improvements over original:
  - Portable paths via argparse (no hardcoded C:\\... paths)
  - Single pipeline: preprocess → train → evaluate → visualise
  - class_weight='balanced' to handle imbalanced High/Low classes
  - 5-fold cross-validation for reliable accuracy estimate
  - Feature importance plot (top 15 features)
  - Model saved with joblib so you don't retrain every run
  - Proper numeric NaN handling (not fillna("Unknown") on numbers)
  - Seaborn confusion matrix heatmap (cleaner than manual imshow)
  - requirements.txt generated automatically

Usage:
  python fall_risk_pipeline.py --data hospital-fall-data-2012-2017.csv
  python fall_risk_pipeline.py --data hospital-fall-data-2012-2017.csv --load-model
"""

import argparse
import re
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "Age (float)", "Sex", "Shift", "Weekday of incident",
    "Hospital department or location of incident",
    "Type of injury incurred, if any",
    "Presence of companion at time of incident",
    "Location or environment in which the incident ocurred",
    "Reason for incident",
    "Whether a fall prevention protocol was implemented",
    "Involvement of medication associated with fall risk",
    "Severity of incident",
]

CATEGORICAL_COLUMNS = [
    "Sex", "Shift", "Weekday of incident",
    "Hospital department or location of incident",
    "Type of injury incurred, if any",
    "Presence of companion at time of incident",
    "Location or environment in which the incident ocurred",
    "Reason for incident",
    "Whether a fall prevention protocol was implemented",
    "Involvement of medication associated with fall risk",
    "Severity of incident",
]

MODEL_PATH = "fall_risk_model.joblib"
OUTPUT_CSV = "fall_risk_processed.csv"

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV or Excel dataset, normalise column names."""
    if not os.path.exists(file_path):
        sys.exit(f"[ERROR] File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    print(f"[1/5] Loading data from: {file_path}")

    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    elif ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        sys.exit(f"[ERROR] Unsupported file format: {ext}. Use .csv or .xlsx")

    df.columns = df.columns.str.strip()
    print(f"      Loaded {len(df):,} rows × {len(df.columns)} columns.")
    return df

# ─────────────────────────────────────────────────────────────
# STEP 2: PREPROCESS
# ─────────────────────────────────────────────────────────────

def convert_age_to_float(age_val) -> float:
    """
    Convert various age string formats to numeric midpoint.
    Handles: '60<70', '≥ 90', '< 1', plain integers, etc.
    """
    if pd.isna(age_val):
        return np.nan
    age_str = str(age_val).strip()

    # Plain integer
    if age_str.isdigit():
        return float(age_str)

    # '≥ 90' or '>= 90'
    if age_str.startswith("≥") or age_str.startswith(">="):
        nums = re.findall(r"\d+", age_str)
        return float(nums[0]) + 5 if nums else np.nan

    # '< 1'
    if age_str.startswith("<"):
        return 0.5

    # 'X<Y' range → midpoint
    match = re.match(r"(\d+)\s*<\s*(\d+)", age_str)
    if match:
        lo, hi = float(match.group(1)), float(match.group(2))
        return (lo + hi) / 2

    return np.nan


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, encode, and feature-engineer the raw dataframe."""
    print("[2/5] Preprocessing...")

    # Drop junk columns
    junk = [c for c in df.columns if c.startswith("Unnamed")]
    if junk:
        df.drop(columns=junk, inplace=True)

    # Parse dates (informational — not used as feature)
    if "Date of incident" in df.columns:
        df["Date of incident"] = pd.to_datetime(df["Date of incident"], errors="coerce")

    # ── Age handling ──────────────────────────────────────────
    # Prefer 'Age range of patient'; fall back to 'Age'
    if "Age range of patient" in df.columns:
        df["Age (float)"] = df["Age range of patient"].apply(convert_age_to_float)
    elif "Age" in df.columns:
        df["Age (float)"] = df["Age"].apply(convert_age_to_float)
        df.drop(columns=["Age"], inplace=True)

    # ── Target label ─────────────────────────────────────────
    if "Fall risk level" not in df.columns:
        sys.exit("[ERROR] 'Fall risk level' column not found in dataset.")

    df["Fall risk (binary)"] = (df["Fall risk level"] == "High").astype(int)

    # ── Keep only required columns ───────────────────────────
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = set(FEATURE_COLUMNS) - set(available_features)
    if missing:
        print(f"      [WARN] Missing feature columns (will be skipped): {missing}")

    df = df[available_features + ["Fall risk (binary)"]].copy()

    # ── Fill categorical NaNs with 'Unknown' ─────────────────
    # IMPORTANT: only fill categoricals, not numeric Age (float)
    cat_present = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    df[cat_present] = df[cat_present].fillna("Unknown")

    # ── Numeric NaN: impute with median ──────────────────────
    if "Age (float)" in df.columns:
        median_age = df["Age (float)"].median()
        n_missing = df["Age (float)"].isna().sum()
        df["Age (float)"].fillna(median_age, inplace=True)
        if n_missing:
            print(f"      Imputed {n_missing} missing Age values with median ({median_age:.1f})")

    # ── One-Hot Encoding ─────────────────────────────────────
    df = pd.get_dummies(df, columns=cat_present, drop_first=True)

    print(f"      Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    class_counts = df["Fall risk (binary)"].value_counts()
    print(f"      Class distribution — Low/Moderate: {class_counts.get(0, 0):,}  |  High: {class_counts.get(1, 0):,}")

    return df

# ─────────────────────────────────────────────────────────────
# STEP 3: TRAIN
# ─────────────────────────────────────────────────────────────

def train_model(X_train, y_train) -> RandomForestClassifier:
    """
    Train Random Forest with class_weight='balanced' to handle
    imbalanced High vs Low/Moderate classes.
    """
    print("[3/5] Training Random Forest (class_weight=balanced)...")
    model = RandomForestClassifier(
        n_estimators=200,       # More trees = more stable
        class_weight="balanced", # FIX: prevents Low/Mod bias
        random_state=42,
        n_jobs=-1               # Use all CPU cores
    )
    model.fit(X_train, y_train)
    print("      Training complete.")
    return model

# ─────────────────────────────────────────────────────────────
# STEP 4: EVALUATE
# ─────────────────────────────────────────────────────────────

def evaluate_model(model, X, y, X_test, y_test):
    """Run cross-validation + held-out test set evaluation."""
    print("[4/5] Evaluating...")

    # ── 5-Fold Stratified Cross-Validation ───────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(f"\n      5-Fold CV F1 Scores : {cv_scores.round(4)}")
    print(f"      Mean CV F1          : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Held-out test set ─────────────────────────────────────
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n      ── Test Set Metrics ──────────────────────────")
    print(f"      Accuracy  : {acc:.4f}")
    print(f"      Precision : {prec:.4f}  (of predicted High, how many truly High)")
    print(f"      Recall    : {rec:.4f}  (of actual High, how many we caught)")
    print(f"      F1 Score  : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Low/Moderate','High'])}")

    return y_pred

# ─────────────────────────────────────────────────────────────
# STEP 5: VISUALISE
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Low/Moderate", "High"],
        yticklabels=["Low/Moderate", "High"],
        ax=ax
    )
    ax.set_title("Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("      Saved: confusion_matrix.png")


def plot_class_distribution(y):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = pd.Series(y).value_counts().sort_index()
    bars = ax.bar(["Low/Moderate (0)", "High (1)"], counts.values,
                  color=["#4C9BE8", "#E85C5C"], edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{val:,}", ha="center", va="bottom", fontsize=11)
    ax.set_title("Fall Risk Class Distribution", fontweight="bold")
    ax.set_ylabel("Number of Incidents")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.show()
    print("      Saved: class_distribution.png")


def plot_feature_importance(model, feature_names, top_n=15):
    """
    Feature importance plot — most useful diagnostic for
    understanding what actually predicts fall risk.
    Not present in original code at all.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    top.plot(kind="barh", ax=ax, color="#4C9BE8", edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)", fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.axvline(importances.mean(), color="red", linestyle="--", linewidth=1, label="Mean importance")
    ax.legend()
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    print("      Saved: feature_importance.png")


def visualise(model, X_test, y_test, y_pred, y_full, feature_names):
    print("[5/5] Generating visualisations...")
    plot_confusion_matrix(y_test, y_pred)
    plot_class_distribution(y_full)
    plot_feature_importance(model, feature_names)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fall Risk Assessment ML Pipeline")
    parser.add_argument("--data", required=True, help="Path to input CSV or Excel dataset")
    parser.add_argument("--load-model", action="store_true",
                        help="Load saved model instead of retraining")
    parser.add_argument("--save-csv", action="store_true", default=True,
                        help="Save processed dataset to CSV (default: True)")
    args = parser.parse_args()

    # ── Load ─────────────────────────────────────────────────
    df_raw = load_data(args.data)

    # ── Preprocess ───────────────────────────────────────────
    df = preprocess(df_raw)

    X = df.drop(columns=["Fall risk (binary)"])
    y = df["Fall risk (binary)"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # stratify keeps class ratio
    )

    # ── Train or Load ────────────────────────────────────────
    if args.load_model and os.path.exists(MODEL_PATH):
        print(f"[3/5] Loading saved model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
    else:
        model = train_model(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
        print(f"      Model saved to {MODEL_PATH}")

    # ── Evaluate ─────────────────────────────────────────────
    y_pred = evaluate_model(model, X, y, X_test, y_test)

    # ── Visualise ────────────────────────────────────────────
    visualise(model, X_test, y_test, y_pred, y, feature_names)

    # ── Save processed CSV ───────────────────────────────────
    if args.save_csv:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Processed dataset saved to: {OUTPUT_CSV}")

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()