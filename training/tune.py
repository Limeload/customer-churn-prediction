"""
Enhanced hyperparameter tuning with RandomizedSearchCV.
Wider parameter distributions, more iterations than initial GridSearch.
Replaces model pkl files with the best-found estimators.

Usage:
  python training/tune.py                          # defaults
  python training/tune.py --n-iter 10 --cv-folds 3 # fast smoke-test
  python training/tune.py --models XGBoost SVM     # subset of models
  python training/tune.py --verbose 1              # per-iteration logging
"""
import argparse
import pickle
import time
from pathlib import Path

import pandas as pd
import numpy as np
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

try:
    from tqdm import tqdm as _tqdm
    def _progress(items, **kw):
        return _tqdm(list(items), **kw)
except ImportError:
    def _progress(items, **kw):
        return items

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MDIR = ROOT / "models" / "bank"

_ALL_MODELS = ["XGBoost", "Random Forest", "Gradient Boosting", "Decision Tree", "SVM", "KNN"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hyperparameter tuning for ChurnGuard bank models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-iter",   type=int, default=30,  metavar="N",
                   help="RandomizedSearchCV candidates per model")
    p.add_argument("--cv-folds", type=int, default=5,   metavar="K",
                   help="Stratified cross-validation folds")
    p.add_argument("--jobs",     type=int, default=-1,  metavar="N",
                   help="Parallel jobs (-1 = all CPUs)")
    p.add_argument("--verbose",  type=int, default=0,   choices=[0, 1, 2, 3],
                   help="Sklearn verbosity level passed to RandomizedSearchCV")
    p.add_argument("--models",   nargs="+", default=None, metavar="NAME",
                   choices=_ALL_MODELS,
                   help="Subset of models to tune (default: all)")
    return p.parse_args()


args   = _parse_args()
N_ITER = args.n_iter
CV     = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

print(f"Config: n_iter={N_ITER}, cv_folds={args.cv_folds}, jobs={args.jobs}, verbose={args.verbose}")
if args.models:
    print(f"Tuning subset: {args.models}")

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
to_tune = {k: v for k, v in spaces.items() if not args.models or k in args.models}
print(f"\n── RandomizedSearchCV (n_iter={N_ITER}, cv={args.cv_folds}) ─────────────")
best_estimators = {}
t_total = time.perf_counter()

for name, (estimator, param_dist, Xtr, Xte) in _progress(
    to_tune.items(), desc="Tuning models", unit="model", leave=True
):
    t0 = time.perf_counter()
    print(f"\n  [{name}] starting search (n_iter={N_ITER}) …", flush=True)
    search = RandomizedSearchCV(
        estimator, param_dist,
        n_iter=N_ITER, cv=CV, scoring="roc_auc",
        n_jobs=args.jobs, random_state=42, verbose=args.verbose,
    )
    search.fit(Xtr, y_train)
    elapsed = time.perf_counter() - t0
    best_estimators[name] = (search.best_estimator_, Xte)
    print(f"  [{name}] cv_auc={search.best_score_:.4f}  elapsed={elapsed:.1f}s")
    print(f"  [{name}] best_params={search.best_params_}")

# ── Results after tuning ──────────────────────────────────────────────────────
print("\n── After tuning ─────────────────────────────────────────────────────")
for name, (model, Xte) in best_estimators.items():
    results_after[name] = report(name, model, Xte, tag="tuned")

# ── Rebuild Stacking with newly tuned base models (only when all three are present)
_stack_bases = {"XGBoost", "Random Forest", "Gradient Boosting"}
if _stack_bases.issubset(best_estimators):
    print("\n  Rebuilding Stacking ensemble with tuned base models…")
    t0 = time.perf_counter()
    stacking = StackingClassifier(
        estimators=[
            ("xgb", best_estimators["XGBoost"][0]),
            ("rf",  best_estimators["Random Forest"][0]),
            ("gbm", best_estimators["Gradient Boosting"][0]),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3, passthrough=False, n_jobs=args.jobs,
    )
    stacking.fit(X_train, y_train)
    print(f"  [Stacking] elapsed={time.perf_counter() - t0:.1f}s")
    results_after["Stacking"] = report("Stacking", stacking, X_test, tag="tuned")
else:
    stacking = None
    print("\n  Skipping Stacking rebuild (not all base models were tuned).")

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
print(f"\nTotal tuning time: {time.perf_counter() - t_total:.1f}s")

# ── Save improved models ──────────────────────────────────────────────────────
print("\nSaving tuned models…")
_filename_map = {
    "XGBoost":           "xgboost_model.pkl",
    "Random Forest":     "random_forest_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Decision Tree":     "decision_tree_model.pkl",
    "SVM":               "svm_model.pkl",
    "KNN":               "knn_model.pkl",
}
saves = {_filename_map[n]: est for n, (est, _) in best_estimators.items()}
if stacking:
    saves["stacking_model.pkl"] = stacking
saves["scaler.pkl"] = scaler

for fname, obj in saves.items():
    with open(MDIR / fname, "wb") as f:
        pickle.dump(obj, f)
    print(f"  saved {fname}")
print(f"Done. {len(saves)} files written to {MDIR}")
