"""Train ML models on Telco Customer Churn dataset and compare with bank dataset."""
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MDIR = ROOT / "models" / "telco"
MDIR.mkdir(parents=True, exist_ok=True)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

np.random.seed(42)

# ── Load & clean ──────────────────────────────────────────────────────────────
print("Loading Telco data...")
df = pd.read_csv(DATA / "telco_churn.csv")
df = df.drop(columns=["customerID"])

# TotalCharges is stored as string and has 11 blank entries
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

# Binary yes/no columns
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
for col in binary_cols:
    df[col] = (df[col] == "Yes").astype(int)

df["gender"] = (df["gender"] == "Male").astype(int)

# Multi-class categoricals → one-hot (drop_first avoids collinearity)
cat_cols = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ── Feature engineering ───────────────────────────────────────────────────────
df["ChargesPerTenure"]    = df["MonthlyCharges"] / (df["tenure"] + 1)
df["TotalToMonthlyRatio"] = df["TotalCharges"]   / (df["MonthlyCharges"] + 1)

# Count add-on services (True = paying for that service)
addon_dummies = [c for c in df.columns if any(
    s in c for s in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                      "TechSupport", "StreamingTV", "StreamingMovies"]
)]
df["ServiceCount"] = df[addon_dummies].sum(axis=1)

print(f"Final shape: {df.shape}")

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

SCALED = {"SVM", "KNN", "Logistic Regression"}

results = {}

def train_eval(name, model, Xtr, Xte, yte):
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    acc = accuracy_score(yte, preds)
    auc = roc_auc_score(yte, proba)
    f1  = f1_score(yte, preds)
    results[name] = {"acc": acc, "auc": auc, "f1": f1}
    print(f"  {name:30s} | Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")
    return model

# ── Base models ───────────────────────────────────────────────────────────────
print("\nBase models (no tuning):")
dt  = train_eval("Decision Tree",       DecisionTreeClassifier(random_state=42), X_train,        X_test,       y_test)
svm = train_eval("SVM",                 SVC(probability=True, random_state=42),  X_train_scaled, X_test_scaled, y_test)
knn = train_eval("KNN",                 KNeighborsClassifier(),                  X_train_scaled, X_test_scaled, y_test)
lr  = train_eval("Logistic Regression", LogisticRegression(max_iter=1000),       X_train_scaled, X_test_scaled, y_test)

# ── Tuned models ──────────────────────────────────────────────────────────────
print("\nTuned models:")

print("  Tuning XGBoost...")
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss"),
    {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1],
     "subsample": [0.8, 1.0]},
    cv=cv, scoring="roc_auc", n_jobs=-1,
)
xgb_grid.fit(X_train, y_train)
print(f"    Best AUC: {xgb_grid.best_score_:.4f} | {xgb_grid.best_params_}")
xgb = train_eval("XGBoost", xgb_grid.best_estimator_, X_train, X_test, y_test)

print("  Tuning Random Forest...")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators": [200, 300], "max_depth": [10, 20, None], "min_samples_split": [2, 5]},
    cv=cv, scoring="roc_auc", n_jobs=-1,
)
rf_grid.fit(X_train, y_train)
print(f"    Best AUC: {rf_grid.best_score_:.4f} | {rf_grid.best_params_}")
rf = train_eval("Random Forest", rf_grid.best_estimator_, X_train, X_test, y_test)

print("  Tuning Gradient Boosting...")
gbm_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 4],
     "subsample": [0.8, 1.0]},
    cv=cv, scoring="roc_auc", n_jobs=-1,
)
gbm_grid.fit(X_train, y_train)
print(f"    Best AUC: {gbm_grid.best_score_:.4f} | {gbm_grid.best_params_}")
gbm = train_eval("Gradient Boosting", gbm_grid.best_estimator_, X_train, X_test, y_test)

# ── Stacking ──────────────────────────────────────────────────────────────────
print("  Training Stacking ensemble...")
stacking = StackingClassifier(
    estimators=[
        ("xgb", xgb_grid.best_estimator_),
        ("rf",  rf_grid.best_estimator_),
        ("gbm", gbm_grid.best_estimator_),
    ],
    final_estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    cv=3, passthrough=False, n_jobs=-1,
)
stacking = train_eval("Stacking", stacking, X_train, X_test, y_test)

# ── Voting ensemble ───────────────────────────────────────────────────────────
print("  Training Voting ensemble...")
voting = VotingClassifier(
    estimators=[
        ("xgb", xgb_grid.best_estimator_),
        ("rf",  rf_grid.best_estimator_),
        ("gbm", gbm_grid.best_estimator_),
    ],
    voting="soft",
)
voting = train_eval("Voting Ensemble", voting, X_train, X_test, y_test)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Model':<30} {'Acc':>7} {'AUC':>7} {'F1':>7}")
print("-"*65)
for name, m in sorted(results.items(), key=lambda x: -x[1]["auc"]):
    print(f"{name:<30} {m['acc']:>7.4f} {m['auc']:>7.4f} {m['f1']:>7.4f}")

best_name = max(results, key=lambda x: results[x]["auc"])
best = results[best_name]
print("="*65)
print(f"\nBest model: {best_name}")
print(f"  Accuracy: {best['acc']:.2%} | AUC: {best['auc']:.4f} | F1: {best['f1']:.4f}")

# ── Compare with bank dataset ─────────────────────────────────────────────────
print("\n" + "="*65)
print("DATASET COMPARISON (best model AUC per dataset)")
print("-"*65)
print(f"  Bank Churn (10K rows, 13 features):   XGBoost AUC = 0.8660")
print(f"  Telco Churn (7K rows, {X.shape[1]} features): {best_name} AUC = {best['auc']:.4f}")

# ── Save telco models ─────────────────────────────────────────────────────────
saves = {
    "xgboost.pkl":           xgb_grid.best_estimator_,
    "random_forest.pkl":     rf_grid.best_estimator_,
    "gradient_boosting.pkl": gbm_grid.best_estimator_,
    "stacking.pkl":          stacking,
    "voting.pkl":            voting,
    "scaler.pkl":            scaler,
}
for name, obj in saves.items():
    with open(MDIR / name, "wb") as f:
        pickle.dump(obj, f)

# ── Save feature schema ───────────────────────────────────────────────────────
import json, sklearn, xgboost as xgb_lib

schema = {
    "feature_columns": list(X.columns),
    "n_features": int(X.shape[1]),
    "trained_with": {
        "scikit-learn": sklearn.__version__,
        "xgboost":      xgb_lib.__version__,
        "numpy":        np.__version__,
    },
}
with open(MDIR / "schema.json", "w") as f:
    json.dump(schema, f, indent=2)

print(f"\nTelco models and schema saved to {MDIR}")
