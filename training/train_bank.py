"""Train all 7 bank ML models and save to models/bank/"""
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

ROOT   = Path(__file__).resolve().parent.parent
DATA   = ROOT / "data"
MDIR   = ROOT / "models" / "bank"
MDIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

print("Loading data...")
df = pd.read_csv(DATA / "churn.csv")
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

df["Gender"]    = LabelEncoder().fit_transform(df["Gender"])
df["Geography"] = LabelEncoder().fit_transform(df["Geography"])

df["BalanceSalaryRatio"] = df["Balance"]     / (df["EstimatedSalary"] + 1)
df["TenureAgeRatio"]     = df["Tenure"]      / (df["Age"] + 1)
df["CreditScorePerAge"]  = df["CreditScore"] / (df["Age"] + 1)

X = df.drop(columns=["Exited"])
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def eval_model(name, model, Xte, yte):
    acc = accuracy_score(yte, model.predict(Xte))
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    print(f"  {name:25s} | Acc: {acc:.4f} | AUC: {auc:.4f}")


# ── Base models (no tuning) ──────────────────────────────────────────────────
print("\nTraining base models...")
base_models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM":           SVC(probability=True, random_state=42),
    "KNN":           KNeighborsClassifier(),
}
SCALED = {"SVM", "KNN"}
for name, model in base_models.items():
    Xtr = X_train_scaled if name in SCALED else X_train
    Xte = X_test_scaled  if name in SCALED else X_test
    model.fit(Xtr, y_train)
    eval_model(name, model, Xte, y_test)

# ── Tuned models ─────────────────────────────────────────────────────────────
print("\nTuning XGBoost...")
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False),
    {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
    cv=cv, scoring="roc_auc", n_jobs=-1,
)
xgb_grid.fit(X_train, y_train)
print(f"  Best AUC: {xgb_grid.best_score_:.4f} | {xgb_grid.best_params_}")
eval_model("XGBoost (tuned)", xgb_grid.best_estimator_, X_test, y_test)

print("\nTuning Random Forest...")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators": [100, 200], "max_depth": [None, 10], "min_samples_split": [2, 5]},
    cv=cv, scoring="roc_auc", n_jobs=-1,
)
rf_grid.fit(X_train, y_train)
print(f"  Best AUC: {rf_grid.best_score_:.4f} | {rf_grid.best_params_}")
eval_model("Random Forest (tuned)", rf_grid.best_estimator_, X_test, y_test)

print("\nTuning Gradient Boosting...")
gbm_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 4]},
    cv=cv, scoring="roc_auc", n_jobs=-1,
)
gbm_grid.fit(X_train, y_train)
print(f"  Best AUC: {gbm_grid.best_score_:.4f} | {gbm_grid.best_params_}")
eval_model("Gradient Boosting (tuned)", gbm_grid.best_estimator_, X_test, y_test)

# ── Stacking ensemble ─────────────────────────────────────────────────────────
print("\nTraining Stacking ensemble...")
stacking = StackingClassifier(
    estimators=[
        ("xgb", xgb_grid.best_estimator_),
        ("rf",  rf_grid.best_estimator_),
        ("gbm", gbm_grid.best_estimator_),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=3,
    passthrough=False,
    n_jobs=-1,
)
stacking.fit(X_train, y_train)
eval_model("Stacking", stacking, X_test, y_test)

# ── Save ─────────────────────────────────────────────────────────────────────
print("\nSaving models...")
saves = {
    "xgboost_model.pkl":           xgb_grid.best_estimator_,
    "random_forest_model.pkl":     rf_grid.best_estimator_,
    "gradient_boosting_model.pkl": gbm_grid.best_estimator_,
    "stacking_model.pkl":          stacking,
    "decision_tree_model.pkl":     base_models["Decision Tree"],
    "svm_model.pkl":               base_models["SVM"],
    "knn_model.pkl":               base_models["KNN"],
    "scaler.pkl":                  scaler,
}
for name, obj in saves.items():
    with open(MDIR / name, "wb") as f:
        pickle.dump(obj, f)

print(f"All models saved to {MDIR}")
