# Task (b): Predict Event_NP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# -------------------------------------------------
# 1) Load + merge (Task a)
# -------------------------------------------------
dep = pd.read_csv("Dependent (1).csv", sep=";", decimal=",")
exp = pd.read_csv("Explanatory (1).csv", sep=";", decimal=",")

df = pd.concat([dep, exp], axis=1)

# -------------------------------------------------
# 2) Target / predictors
# -------------------------------------------------
y = pd.to_numeric(df["Event_NP"], errors="coerce").astype(int)
X = df.drop(columns=["Event_NP"])

# Optional: mirror your R cleaning choice (drop UzletiKedv)
if "UzletiKedv" in X.columns:
    X = X.drop(columns=["UzletiKedv"])

# -------------------------------------------------
# 3) Column types
# -------------------------------------------------
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

# -------------------------------------------------
# 4) Preprocessing
# -------------------------------------------------
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# -------------------------------------------------
# 5) Train / test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# 6) Models (reasonable set)
# -------------------------------------------------
models = {
    "Logistic": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),
    "LassoLogistic": LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=3000,
        class_weight="balanced"
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),
    "GradientBoosting": GradientBoostingClassifier(
        random_state=42
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        random_state=42
    )
}

# -------------------------------------------------
# 7) Fit + evaluate
# -------------------------------------------------
results = []

print("\nMODEL COMPARISON (Task b)")
print("=" * 60)

for name, model in models.items():
    pipe = Pipeline([
        ("prep", preprocess),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, pred)

    print(f"\n{name}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))

    results.append((name, auc, f1))

# -------------------------------------------------
# 8) Summary
# -------------------------------------------------
summary = pd.DataFrame(results, columns=["Model", "ROC_AUC", "F1"])
summary = summary.sort_values("ROC_AUC", ascending=False)

print("\nSUMMARY TABLE")
print("=" * 60)
print(summary.to_string(index=False))


summary = summary.sort_values("ROC_AUC")

plt.figure(figsize=(7,4))
plt.barh(summary["Model"], summary["ROC_AUC"])
plt.xlabel("ROC-AUC")
plt.title("Model comparison – ROC-AUC")
plt.xlim(0.9, 1.0)
plt.tight_layout()
plt.show()
