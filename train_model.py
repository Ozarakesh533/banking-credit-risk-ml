"""
Project 2: Customer Credit Risk Analysis & ML Dashboard
Barclays Data Analyst Role — JD Aligned
Author: Rakesh Oza
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, roc_curve)
from sklearn.pipeline import Pipeline
import pickle, os, json
from datetime import datetime

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
np.random.seed(42)

# ─────────────────────────────────────────────
# GENERATE SYNTHETIC CREDIT DATASET
# (mirrors Kaggle Give Me Some Credit structure)
# ─────────────────────────────────────────────
def generate_credit_data(n=2000):
    age            = np.random.randint(22, 75, n)
    income         = np.random.exponential(40000, n).clip(8000, 500000)
    debt_ratio     = np.random.beta(2, 5, n)
    credit_lines   = np.random.randint(1, 20, n)
    missed_30_60   = np.random.poisson(0.3, n)
    missed_60_90   = np.random.poisson(0.1, n)
    missed_90_plus = np.random.poisson(0.05, n)
    revolving_util = np.random.beta(3, 5, n)
    dependents     = np.random.randint(0, 5, n)
    loan_amount    = np.random.exponential(150000, n).clip(10000, 2000000)
    employment_yrs = np.random.randint(0, 35, n)

    # Default probability based on features (business logic)
    default_prob = (
        0.05
        + 0.15 * (missed_90_plus > 0)
        + 0.10 * (missed_60_90 > 0)
        + 0.05 * (missed_30_60 > 1)
        + 0.08 * (debt_ratio > 0.6)
        + 0.05 * (revolving_util > 0.7)
        - 0.03 * (employment_yrs > 10)
        - 0.02 * (age > 40)
        + 0.04 * (income < 20000)
    ).clip(0.01, 0.95)

    default = (np.random.rand(n) < default_prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": income.round(2),
        "debt_ratio": debt_ratio.round(4),
        "credit_lines_open": credit_lines,
        "missed_pmts_30_60": missed_30_60,
        "missed_pmts_60_90": missed_60_90,
        "missed_pmts_90plus": missed_90_plus,
        "revolving_utilization": revolving_util.round(4),
        "number_of_dependents": dependents,
        "loan_amount": loan_amount.round(2),
        "employment_years": employment_yrs,
        "defaulted": default,
    })

    # Introduce some nulls for realism
    df.loc[np.random.choice(df.index, 40, replace=False), "annual_income"] = np.nan
    df.loc[np.random.choice(df.index, 25, replace=False), "employment_years"] = np.nan

    return df


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df["annual_income"].fillna(df["annual_income"].median(), inplace=True)
    df["employment_years"].fillna(df["employment_years"].median(), inplace=True)

    # Feature engineering
    df["total_missed_payments"] = df["missed_pmts_30_60"] + df["missed_pmts_60_90"] + df["missed_pmts_90plus"]
    df["risk_score_raw"]  = (
        df["debt_ratio"] * 0.3 +
        df["revolving_utilization"] * 0.25 +
        (df["total_missed_payments"] / 10).clip(0,1) * 0.45
    )
    df["income_to_loan"]  = (df["annual_income"] / df["loan_amount"]).round(4)

    # Risk segment labels (business rule)
    conditions = [
        df["risk_score_raw"] < 0.2,
        (df["risk_score_raw"] >= 0.2) & (df["risk_score_raw"] < 0.45),
        df["risk_score_raw"] >= 0.45,
    ]
    df["risk_segment"] = np.select(conditions, ["LOW", "MEDIUM", "HIGH"])
    return df


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
FEATURES = [
    "age","annual_income","debt_ratio","credit_lines_open",
    "missed_pmts_30_60","missed_pmts_60_90","missed_pmts_90plus",
    "revolving_utilization","number_of_dependents","loan_amount",
    "employment_years","total_missed_payments","risk_score_raw","income_to_loan"
]

def train_models(df):
    X = df[FEATURES]
    y = df["defaulted"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        auc    = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            "model": model, "auc": round(auc,4),
            "precision": round(report["1"]["precision"],4),
            "recall":    round(report["1"]["recall"],4),
            "f1":        round(report["1"]["f1-score"],4),
            "y_test": y_test, "y_prob": y_prob, "y_pred": y_pred,
            "X_test": X_test,
        }
        print(f"  {name:25s} | AUC: {auc:.4f} | F1: {report['1']['f1-score']:.4f}")

    # Save best model (highest AUC)
    best_name  = max(results, key=lambda k: results[k]["auc"])
    best_model = results[best_name]["model"]
    with open("models/best_model.pkl","wb") as f:
        pickle.dump({"model": best_model, "features": FEATURES, "name": best_name}, f)
    print(f"\n✅ Best model saved: {best_name} (AUC={results[best_name]['auc']})")

    # Save feature importance (Random Forest)
    rf = results["Random Forest"]["model"]
    fi = pd.DataFrame({
        "feature": FEATURES,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    fi.to_csv("data/feature_importance.csv", index=False)

    return results, X_test, y_test


if __name__ == "__main__":
    print("📦 Generating credit dataset...")
    df_raw = generate_credit_data(2000)
    df     = preprocess(df_raw)
    df.to_csv("data/credit_data.csv", index=False)
    print(f"   Dataset: {len(df)} rows | Default rate: {df['defaulted'].mean():.1%}")

    print("\n🤖 Training models...")
    results, X_test, y_test = train_models(df)

    print("\n=== MODEL COMPARISON ===")
    comparison = pd.DataFrame([{
        "Model": name, "AUC": r["auc"],
        "Precision": r["precision"], "Recall": r["recall"], "F1": r["f1"]
    } for name, r in results.items()])
    print(comparison.to_string(index=False))
    comparison.to_csv("data/model_comparison.csv", index=False)
