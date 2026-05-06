"""
model_training.py
=================
End-to-End ML Pipeline for CTG Fetal Health Classification
Covers Steps 3.1 – 3.7 from the project brief:
  3.1  Load Data & Import Libraries
  3.2  Preprocessing, EDA, Train-Test Split
  3.3  Model Training (3 base models)
  3.4  Model Evaluation (classification_report + K-Fold / Stratified K-Fold)
  3.5  Testing on New Instances
  3.6  Hyperparameter Tuning (RandomizedSearchCV)
  3.7  Model Serialization (joblib)
"""

# ── 3.1  IMPORT LIBRARIES ────────────────────────────────────────────────────
import json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")
np.random.seed(42)

print("=" * 60)
print("  CTG Fetal Health Classification  –  Full ML Pipeline")
print("=" * 60)

# ── 3.1  LOAD DATA ────────────────────────────────────────────────────────────
# The real dataset is CTG.xlsx from UCI. Here we generate a representative
# synthetic version so the script is self-contained.  Replace the block
# below with:   df = pd.read_excel("CTG.xlsx", sheet_name="Raw Data")
# when you have the real file.

def generate_ctg_data(n=2126, seed=42):
    rng = np.random.default_rng(seed)
    # Approximate distributions from the UCI dataset description
    nsp = rng.choice([1, 2, 3], size=n, p=[0.778, 0.136, 0.086])
    # Feature means shift slightly per class
    shift = {1: 0.0, 2: 0.5, 3: 1.2}
    rows = []
    for label in nsp:
        s = shift[label]
        rows.append({
            "LB":        rng.normal(133 + s * 2,   9),
            "AC":        max(0, rng.normal(0.003 + s * 0.001, 0.003)),
            "FM":        max(0, rng.normal(0.09 + s * 0.01,  0.18)),
            "UC":        max(0, rng.normal(0.004 + s * 0.001, 0.003)),
            "ASTV":      rng.normal(46 + s * 5,  13),
            "MSTV":      rng.normal(1.33 + s * 0.2, 0.9),
            "ALTV":      rng.normal(9.8 + s * 4,  14),
            "MLTV":      rng.normal(8.2 + s * 1,   5),
            "Width":     rng.normal(70 + s * 5,   39),
            "Min":       rng.normal(93 + s * 3,   28),
            "Max":       rng.normal(164 + s * 3,  25),
            "Nmax":      int(max(1, rng.normal(4 + s, 2))),
            "Nzeros":    int(max(0, rng.normal(0.3 + s * 0.2, 0.7))),
            "Mode":      rng.normal(137 + s * 2,  16),
            "Mean":      rng.normal(134 + s * 2,  16),
            "Median":    rng.normal(138 + s * 2,  16),
            "Variance":  max(0, rng.normal(18 + s * 5, 28)),
            "Tendency":  rng.choice([-1, 0, 1], p=[0.15, 0.73, 0.12]),
            "NSP":       label,
        })
    return pd.DataFrame(rows)

df = generate_ctg_data()
print(f"\n[3.1] Dataset loaded  –  shape: {df.shape}")
print(df.head(3).to_string())

# ── 3.2  PREPROCESSING, EDA & TRAIN-TEST SPLIT ────────────────────────────────
print("\n[3.2] Preprocessing & EDA")

# Missing values
print(f"  Missing values: {df.isnull().sum().sum()}")

# Class distribution
print("\n  Class distribution (NSP):")
print(df["NSP"].value_counts().to_string())

# Descriptive stats
print("\n  Descriptive stats:")
print(df.describe().round(2).to_string())

# ── Outlier handling: IQR-based capping (Winsorisation) ─────────────────────
features = [c for c in df.columns if c != "NSP"]
df_clean = df.copy()
for col in features:
    Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df_clean[col] = df_clean[col].clip(lower, upper)
print("\n  Outliers capped via IQR Winsorisation.")

# ── Correlation heatmap saved to file ────────────────────────────────────────
plt.figure(figsize=(14, 10))
corr = df_clean[features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", linewidths=0.3)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=120)
plt.close()
print("  Correlation heatmap saved → correlation_heatmap.png")

# ── Remove highly correlated features (|r| > 0.90) ──────────────────────────
corr_matrix = corr.abs()
upper_tri   = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop     = [col for col in upper_tri.columns if any(upper_tri[col] > 0.90)]
print(f"  Dropping highly correlated features: {to_drop if to_drop else 'none'}")
df_clean.drop(columns=to_drop, inplace=True)
features = [c for c in df_clean.columns if c != "NSP"]

# ── Train-Test Split (80/20, stratified) ─────────────────────────────────────
X = df_clean[features].values
y = df_clean["NSP"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n  Train size: {X_train.shape[0]}   Test size: {X_test.shape[0]}")

# ── 3.3  MODEL TRAINING ───────────────────────────────────────────────────────
print("\n[3.3] Training base models (default parameters)")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(random_state=42),
    "KNN":                 KNeighborsClassifier(),
}

base_results = {}
for name, model in base_models.items():
    model.fit(X_train_sc, y_train)
    acc = model.score(X_test_sc, y_test)
    base_results[name] = acc
    print(f"  {name:<25} Accuracy: {acc:.4f}")

# ── 3.4  MODEL EVALUATION ─────────────────────────────────────────────────────
print("\n[3.4] Detailed evaluation")

target_names = ["Normal", "Suspect", "Pathologic"]
all_reports  = {}

for name, model in base_models.items():
    y_pred = model.predict(X_test_sc)
    report = classification_report(y_test, y_pred,
                                   target_names=target_names, output_dict=True)
    all_reports[name] = report
    print(f"\n  ── {name} ──")
    print(classification_report(y_test, y_pred, target_names=target_names))

# Save classification report
with open("classification_report.json", "w") as f:
    json.dump(all_reports, f, indent=2)
print("  Classification report saved → classification_report.json")

# ── K-Fold & Stratified K-Fold Cross-Validation ──────────────────────────────
print("\n  Cross-Validation (Stratified K-Fold, k=5)")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in base_models.items():
    scores = cross_val_score(model, X_train_sc, y_train,
                             cv=skf, scoring="accuracy")
    print(f"  {name:<25} CV scores: {scores.round(4)}  "
          f"Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")

# ── 3.5  TESTING ON NEW INSTANCES ────────────────────────────────────────────
print("\n[3.5] Inference on 3 new instances (from test set)")
sample_idx    = [0, 10, 20]
sample_X      = X_test_sc[sample_idx]
sample_y_true = y_test[sample_idx]
label_map     = {1: "Normal", 2: "Suspect", 3: "Pathologic"}

for i, (x, true) in enumerate(zip(sample_X, sample_y_true)):
    pred = base_models["Random Forest"].predict([x])[0]
    print(f"  Instance {i+1}:  True={label_map[true]}   "
          f"Predicted={label_map[pred]}")

# ── 3.6  HYPERPARAMETER TUNING ────────────────────────────────────────────────
print("\n[3.6] Hyperparameter tuning (RandomizedSearchCV)")

# Tune Random Forest (best base model)
param_dist = {
    "n_estimators":      [50, 100, 200, 300],
    "max_depth":         [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2"],
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=skf,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
    verbose=0,
)
rf_random.fit(X_train_sc, y_train)
best_rf = rf_random.best_estimator_

tuned_acc = best_rf.score(X_test_sc, y_test)
print(f"  Best params : {rf_random.best_params_}")
print(f"  Tuned RF accuracy on test set: {tuned_acc:.4f}  "
      f"(base: {base_results['Random Forest']:.4f})")

# Confusion matrix for tuned RF
cm = confusion_matrix(y_test, best_rf.predict(X_test_sc))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Tuned Random Forest – Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.close()
print("  Confusion matrix saved → confusion_matrix.png")

# ── 3.7  MODEL SERIALIZATION ──────────────────────────────────────────────────
print("\n[3.7] Serializing best model + scaler")

joblib.dump(best_rf, "rf_best_model.pkl")
joblib.dump(scaler,  "scaler.pkl")
joblib.dump(features, "feature_names.pkl")
print("  Saved: rf_best_model.pkl | scaler.pkl | feature_names.pkl")

# Reload and verify
loaded_model  = joblib.load("rf_best_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")
verify_pred   = loaded_model.predict(loaded_scaler.transform(X_test[:3]))
print(f"  Reload verification predictions: "
      f"{[label_map[p] for p in verify_pred]}")

print("\n✅  Pipeline complete.  All artefacts saved to working directory.\n")
