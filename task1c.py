import numpy as np
import pandas as pd

from lifelines import WeibullAFTFitter
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV


# -----------------------
# 1) Load + merge
# -----------------------
dep = pd.read_csv("Dependent (1).csv", sep=";", decimal=",")
exp = pd.read_csv("Explanatory (1).csv", sep=";", decimal=",")
df = pd.concat([dep, exp], axis=1)

# -----------------------
# 2) Filter: only overdue cancellations
# -----------------------
df["Event_NP"] = pd.to_numeric(df["Event_NP"], errors="coerce").astype(int)
df = df[df["Event_NP"] == 1].copy()

# Time must be positive
df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
df = df[df["Time"].notna() & (df["Time"] > 0)].copy()

# Optional drop 
if "UzletiKedv" in df.columns:
    df = df.drop(columns=["UzletiKedv"])

# -----------------------
# 3) Build X, y
# -----------------------
y = np.log(df["Time"].values)  # for screening only
X = df.drop(columns=["Event_NP", "Time"], errors="ignore")

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

# -----------------------
# 4) Fast screening with Lasso on log(Time)
# -----------------------
Xmat = preprocess.fit_transform(X)

# feature names for interpretability
feature_names = list(num_cols)
if len(cat_cols) > 0:
    oh = preprocess.named_transformers_["cat"].named_steps["oh"]
    feature_names += list(oh.get_feature_names_out(cat_cols))

lasso = LassoCV(cv=5, random_state=42, n_alphas=40, max_iter=20000)
lasso.fit(Xmat, y)

coef = pd.Series(lasso.coef_, index=feature_names)
selected = coef[coef.abs() > 1e-8].index.tolist()

print("\nSelected variables (Lasso screening):")
for v in selected[:50]:
    print(" -", v)
print(f"\nTotal selected: {len(selected)}")

# -----------------------
# 5) Final inference: Weibull AFT on selected vars
# -----------------------
# Build a clean numeric dataframe for lifelines:
df_model = df[["Time"]].copy()

# Add selected features
# For numeric features: directly from df
for col in num_cols:
    if col in selected:
        df_model[col] = pd.to_numeric(df[col], errors="coerce")

# For categorical: add dummies and take selected
if len(cat_cols) > 0:
    dummies = pd.get_dummies(df[cat_cols], drop_first=True)
    for col in dummies.columns:
        # lifelines uses exact column names, so match the same naming as pandas get_dummies
        if col in selected:
            df_model[col] = dummies[col]

df_model = df_model.dropna().reset_index(drop=True)

aft = WeibullAFTFitter()
aft.fit(df_model, duration_col="Time")

out = aft.summary.reset_index()

print("DEBUG columns:", list(out.columns)) # ezt 1x nézd meg

# Case 1: lifelines MultiIndex -> columns like ['param', 'covariate', 'coef', 'se(coef)', 'p', ...]
if "param" in out.columns and "covariate" in out.columns:
    out = out[out["param"] == "lambda_"].copy()
    out["variable"] = out["covariate"].astype(str)

# Case 2: single index -> first column contains terms like 'lambda_XYZ'
else:
    out = out.rename(columns={out.columns[0]: "term"})
    out = out[out["term"].astype(str).str.startswith("lambda_")].copy()
    out["variable"] = out["term"].astype(str).str.replace("lambda_", "", regex=False)

# Ensure p column name
if "p" not in out.columns:
    p_candidates = [c for c in out.columns if c.lower() in ["p", "pvalue", "p-value", "p_value"]]
if len(p_candidates) == 1:
    out = out.rename(columns={p_candidates[0]: "p"})
else:
    raise ValueError(f"Could not find p-value column. Columns: {list(out.columns)}")

out = out.sort_values("p")

print("\nTop determinants by p-value:")
print(out[["variable", "coef", "se(coef)", "p"]].head(20).to_string(index=False))

sig = out[out["p"] < 0.05][["variable", "coef", "se(coef)", "p"]]
sig.sort_values("p").head(20)
sig