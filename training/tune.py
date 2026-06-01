"""
Enhanced hyperparameter tuning with RandomizedSearchCV.
Wider parameter distributions, more iterations than initial GridSearch.
Replaces model pkl files with the best-found estimators.
"""
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MDIR = ROOT / "models" / "bank"
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

np.random.seed(42)
N_ITER = 30       # candidates per model
CV     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading bank churn dataset…")
df = pd.read_csv(DATA / "churn.csv")
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
df["Gender"]    = LabelEncoder().fit_transform(df["Gender"])
df["Geography"] = LabelEncoder().fit_transform(df["Geography"])
df["BalanceSalaryRatio"] = df["Balance"]      / (df["EstimatedSalary"] + 1)
df["TenureAgeRatio"]     = df["Tenure"]       / (df["Age"] + 1)
df["CreditScorePerAge"]  = df["CreditScore"]  / (df["Age"] + 1)

X = df.drop(columns=["Exited"])
y = df["Exited"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

def report(name, model, Xte, tag=""):
    acc = accuracy_score(y_test, model.predict(Xte))
    auc = roc_auc_score(y_test, model.predict_proba(Xte)[:, 1])
    f1  = f1_score(y_test, model.predict(Xte))
    marker = " ✓" if tag == "tuned" else ""
    print(f"  {name:28s} | Acc {acc:.4f} | AUC {auc:.4f} | F1 {f1:.4f}{marker}")
    return {"acc": acc, "auc": auc, "f1": f1}

results_before = {}
results_after  = {}

# ── Baseline (current saved models) ───────────────────────────────────────────
print("\n── Baseline (pre-tuning) ─────────────────────────────────────────────")
saved = {
    "XGBoost":           (MDIR / "xgboost_model.pkl",           X_test),
    "Random Forest":     (MDIR / "random_forest_model.pkl",     X_test),
    "Gradient Boosting": (MDIR / "gradient_boosting_model.pkl", X_test),
    "Decision Tree":     (MDIR / "decision_tree_model.pkl",     X_test),
    "SVM":               (MDIR / "svm_model.pkl",               X_test_s),
    "KNN":               (MDIR / "knn_model.pkl",               X_test_s),
}
for name, (path, Xte) in saved.items():
    with open(path, "rb") as f:
        m = pickle.load(f)
    results_before[name] = report(name, m, Xte)

# ── RandomizedSearchCV parameter spaces ───────────────────────────────────────
spaces = {
    "XGBoost": (
        XGBClassifier(random_state=42, eval_metric="logloss"),
        {
            "n_estimators":      randint(100, 500),
            "max_depth":         randint(2, 8),
            "learning_rate":     loguniform(0.01, 0.3),
            "subsample":         uniform(0.6, 0.4),
            "colsample_bytree":  uniform(0.6, 0.4),
            "gamma":             uniform(0, 0.5),
            "reg_alpha":         loguniform(1e-3, 10),
            "reg_lambda":        loguniform(1e-3, 10),
            "min_child_weight":  randint(1, 6),
        },
        X_train, X_test,
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators":      randint(100, 600),
            "max_depth":         [None, 5, 10, 15, 20],
            "min_samples_split": randint(2, 20),
            "min_samples_leaf":  randint(1, 10),
            "max_features":      ["sqrt", "log2", 0.3, 0.5],
            "bootstrap":         [True, False],
        },
        X_train, X_test,
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators":      randint(100, 400),
            "learning_rate":     loguniform(0.01, 0.3),
            "max_depth":         randint(2, 6),
            "subsample":         uniform(0.6, 0.4),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf":  randint(1, 10),
            "max_features":      ["sqrt", "log2", None],
        },
        X_train, X_test,
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "max_depth":         [None, 5, 10, 15, 20, 25],
            "min_samples_split": randint(2, 30),
            "min_samples_leaf":  randint(1, 15),
            "criterion":         ["gini", "entropy"],
            "splitter":          ["best", "random"],
            "max_features":      ["sqrt", "log2", None],
        },
        X_train, X_test,
    ),
    "SVM": (
        SVC(probability=True, random_state=42),
        {
            "C":       loguniform(0.01, 100),
            "gamma":   loguniform(1e-4, 1),
            "kernel":  ["rbf", "poly", "sigmoid"],
        },
        X_train_s, X_test_s,
    ),
    "KNN": (
        KNeighborsClassifier(n_jobs=-1),
        {
            "n_neighbors": randint(3, 30),
            "weights":     ["uniform", "distance"],
            "metric":      ["euclidean", "manhattan", "minkowski"],
            "leaf_size":   randint(10, 50),
            "p":           [1, 2],
        },
        X_train_s, X_test_s,
    ),
}

# ── Run RandomizedSearchCV ────────────────────────────────────────────────────
print(f"\n── RandomizedSearchCV (n_iter={N_ITER} per model) ──────────────────────")
best_estimators = {}

for name, (estimator, param_dist, Xtr, Xte) in spaces.items():
    print(f"  Tuning {name}…", end=" ", flush=True)
    search = RandomizedSearchCV(
        estimator, param_dist,
        n_iter=N_ITER, cv=CV, scoring="roc_auc",
        n_jobs=-1, random_state=42, verbose=0,
    )
    search.fit(Xtr, y_train)
    best = search.best_estimator_
    best_estimators[name] = (best, Xte)
    print(f"cv_auc={search.best_score_:.4f}  params={search.best_params_}")

# ── Results after tuning ──────────────────────────────────────────────────────
print("\n── After tuning ─────────────────────────────────────────────────────")
for name, (model, Xte) in best_estimators.items():
    results_after[name] = report(name, model, Xte, tag="tuned")

# ── Rebuild Stacking with newly tuned base models ─────────────────────────────
print("\n  Rebuilding Stacking ensemble with tuned base models…")
stacking = StackingClassifier(
    estimators=[
        ("xgb", best_estimators["XGBoost"][0]),
        ("rf",  best_estimators["Random Forest"][0]),
        ("gbm", best_estimators["Gradient Boosting"][0]),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=3, passthrough=False, n_jobs=-1,
)
stacking.fit(X_train, y_train)
results_after["Stacking"] = report("Stacking", stacking, X_test, tag="tuned")

# ── Delta summary ─────────────────────────────────────────────────────────────
print("\n" + "="*72)
print(f"{'Model':<28} {'AUC Before':>11} {'AUC After':>10} {'Δ AUC':>8}")
print("-"*72)
for name in results_before:
    b = results_before[name]["auc"]
    a = results_after[name]["auc"]
    delta = a - b
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
    print(f"  {name:<26} {b:>11.4f} {a:>10.4f} {arrow} {abs(delta):.4f}")
print("="*72)

# ── Save improved models ──────────────────────────────────────────────────────
print("\nSaving tuned models…")
saves = {
    "xgboost_model.pkl":           best_estimators["XGBoost"][0],
    "random_forest_model.pkl":     best_estimators["Random Forest"][0],
    "gradient_boosting_model.pkl": best_estimators["Gradient Boosting"][0],
    "decision_tree_model.pkl":     best_estimators["Decision Tree"][0],
    "svm_model.pkl":               best_estimators["SVM"][0],
    "knn_model.pkl":               best_estimators["KNN"][0],
    "stacking_model.pkl":          stacking,
    "scaler.pkl":                  scaler,
}
for name, obj in saves.items():
    with open(MDIR / name, "wb") as f:
        pickle.dump(obj, f)
print(f"All tuned models saved to {MDIR}")
