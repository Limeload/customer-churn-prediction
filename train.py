"""Train all 5 ML models and save to models/"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

np.random.seed(42)

print("Loading data...")
df = pd.read_csv("churn.csv")
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

le_gender = LabelEncoder()
le_geo = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Geography"] = le_geo.fit_transform(df["Geography"])

df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
df["TenureAgeRatio"] = df["Tenure"] / (df["Age"] + 1)
df["CreditScorePerAge"] = df["CreditScore"] / (df["Age"] + 1)

X = df.drop(columns=["Exited"])
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining base models...")
base_models = {
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
}

SCALED = {"SVM", "KNN"}
for name, model in base_models.items():
    Xtr = X_train_scaled if name in SCALED else X_train
    Xte = X_test_scaled if name in SCALED else X_test
    model.fit(Xtr, y_train)
    acc = accuracy_score(y_test, model.predict(Xte))
    auc = roc_auc_score(y_test, model.predict_proba(Xte)[:, 1])
    print(f"  {name:20s} | Acc: {acc:.4f} | AUC: {auc:.4f}")

print("\nTuning XGBoost...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False),
    {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
    cv=cv, scoring="roc_auc", n_jobs=-1,
)
xgb_grid.fit(X_train, y_train)
print(f"  Best XGBoost AUC: {xgb_grid.best_score_:.4f} | {xgb_grid.best_params_}")

print("Tuning Random Forest...")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators": [100, 200], "max_depth": [None, 10], "min_samples_split": [2, 5]},
    cv=cv, scoring="roc_auc", n_jobs=-1,
)
rf_grid.fit(X_train, y_train)
print(f"  Best RF AUC: {rf_grid.best_score_:.4f} | {rf_grid.best_params_}")

print("\nSaving models...")
os.makedirs("models", exist_ok=True)

with open("models/xgboost_model.pkl", "wb") as f:
    pickle.dump(xgb_grid.best_estimator_, f)
with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_grid.best_estimator_, f)
with open("models/decision_tree_model.pkl", "wb") as f:
    pickle.dump(base_models["Decision Tree"], f)
with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(base_models["SVM"], f)
with open("models/knn_model.pkl", "wb") as f:
    pickle.dump(base_models["KNN"], f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("All models saved to models/")
