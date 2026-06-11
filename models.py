import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    classification_report,
)

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────
#  LOAD PREPROCESSED DATA
# ─────────────────────────────────────────────
print("Loading preprocessed data...")
data           = joblib.load("models/preprocessed_data.joblib")
X_train_sc     = data["X_train_sc"]
X_test_sc      = data["X_test_sc"]
y_train_enc    = data["y_train_enc"]
y_test_enc     = data["y_test_enc"]

print(f"X_train_sc   : {X_train_sc.shape}")
print(f"X_test_sc    : {X_test_sc.shape}")
print(f"Class dist   : 0={(y_train_enc==0).sum()}  1={(y_train_enc==1).sum()}")

# ─────────────────────────────────────────────
#  APPLY SMOTE AND ADASYN
# ─────────────────────────────────────────────
print("\nApplying SMOTE...")
sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train_sc, y_train_enc)
print(f"After SMOTE  : 0={(y_train_smote==0).sum()}  1={(y_train_smote==1).sum()}")

print("Applying ADASYN...")
ad = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = ad.fit_resample(X_train_sc, y_train_enc)
print(f"After ADASYN : 0={(y_train_adasyn==0).sum()}  1={(y_train_adasyn==1).sum()}")

# ─────────────────────────────────────────────
#  HELPER — evaluate any trained model
# ─────────────────────────────────────────────
def evaluate(model_name, variant, model):
    y_pred = model.predict(X_test_sc)
    y_prob = (
        model.predict_proba(X_test_sc)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test_sc)
    )
    return {
        "model":            model_name,
        "variant":          variant,
        "accuracy":         round(accuracy_score (y_test_enc, y_pred), 4),
        "precision":        round(precision_score(y_test_enc, y_pred), 4),
        "recall":           round(recall_score   (y_test_enc, y_pred), 4),
        "f1_score":         round(f1_score       (y_test_enc, y_pred), 4),
        "roc_auc":          round(roc_auc_score  (y_test_enc, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test_enc, y_pred),
        "report":           classification_report(y_test_enc, y_pred),
    }

# ─────────────────────────────────────────────
#  HELPER — measure latency for a trained model
# ─────────────────────────────────────────────
def measure_latency(model, sample, n=1000):
    # warm up
    for _ in range(10):
        model.predict(sample)
    start = time.perf_counter()
    for _ in range(n):
        model.predict(sample)
    end = time.perf_counter()
    return round((end - start) / n * 1000, 4)

# ─────────────────────────────────────────────
#  MODEL DEFINITIONS
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost":             XGBClassifier(n_estimators=100, random_state=42,
                                         eval_metric="logloss", verbosity=0, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost":            AdaBoostClassifier(n_estimators=100, random_state=42),
}

variants = {
    "Base":   (X_train_sc,     y_train_enc),
    "SMOTE":  (X_train_smote,  y_train_smote),
    "ADASYN": (X_train_adasyn, y_train_adasyn),
}

single_sample = X_test_sc[:1]

# ─────────────────────────────────────────────
#  TRAIN ALL MODELS × ALL VARIANTS
# ─────────────────────────────────────────────
all_metrics    = []
trained_models = {}
latency_results = {}

print("\n" + "="*65)
print("   TRAINING ALL MODELS  —  BASE | SMOTE | ADASYN")
print("="*65)

for model_name, model_obj in models.items():
    print(f"\n── {model_name} ──")
    for variant, (X_tr, y_tr) in variants.items():
        m = clone(model_obj)
        m.fit(X_tr, y_tr)

        metrics = evaluate(model_name, variant, m)
        lat_ms  = measure_latency(m, single_sample)

        metrics["latency_ms"] = lat_ms
        all_metrics.append(metrics)

        key = f"{model_name}_{variant}"
        trained_models[key] = m
        latency_results[key] = lat_ms

        print(f"  {variant:<8}  "
              f"Acc={metrics['accuracy']:.4f}  "
              f"Recall={metrics['recall']:.4f}  "
              f"F1={metrics['f1_score']:.4f}  "
              f"ROC-AUC={metrics['roc_auc']:.4f}  "
              f"Latency={lat_ms} ms")

# ─────────────────────────────────────────────
#  COMPARISON TABLE
# ─────────────────────────────────────────────
metrics_df = pd.DataFrame([{
    "Model":       m["model"],
    "Variant":     m["variant"],
    "Accuracy":    m["accuracy"],
    "Precision":   m["precision"],
    "Recall":      m["recall"],
    "F1 Score":    m["f1_score"],
    "ROC-AUC":     m["roc_auc"],
    "Latency (ms)":m["latency_ms"],
} for m in all_metrics])

print("\n" + "="*65)
print("   FULL COMPARISON  (sorted by Recall — business priority)")
print("="*65)
print(metrics_df.sort_values("Recall", ascending=False).to_string(index=False))

# ─────────────────────────────────────────────
#  SAVE ALL TRAINED MODELS
# ─────────────────────────────────────────────
print("\n" + "="*65)
print("   SAVING ALL MODELS")
print("="*65)

for key, m in trained_models.items():
    fname = key.replace(" ", "_").lower() + ".joblib"
    joblib.dump(m, f"models/{fname}")
    print(f"  Saved → models/{fname}")

joblib.dump(all_metrics,    "models/all_metrics.joblib")
joblib.dump(latency_results,"models/latency_results.joblib")
print("  Saved → models/all_metrics.joblib")
print("  Saved → models/latency_results.joblib")

# ═════════════════════════════════════════════
#  FINE TUNE LR ADASYN
#  Business reason: Highest Recall (0.8995)
#  Minimize False Negatives — don't lose customers
# ═════════════════════════════════════════════
print("\n" + "="*65)
print("   HYPERPARAMETER TUNING — LR ADASYN")
print("   Business reason: Minimize False Negatives")
print("="*65)

lr_param_grid = {
    "C":        [0.001, 0.01, 0.1, 1, 10, 100],
    "solver":   ["lbfgs", "saga"],
    "max_iter": [1000, 2000, 3000],
    "tol":      [1e-4, 1e-3, 1e-2],
}

rs_cv = RandomizedSearchCV(
    estimator           = LogisticRegression(random_state=42),
    param_distributions = lr_param_grid,
    n_iter              = 50,
    scoring             = "recall",      # tuning for recall — business priority
    cv                  = 5,
    n_jobs              = -1,
    random_state        = 42,
    verbose             = 1,
)
rs_cv.fit(X_train_adasyn, y_train_adasyn)

lr_adasyn_tuned  = rs_cv.best_estimator_
tuned_metrics    = evaluate("Logistic Regression", "ADASYN + Tuned", lr_adasyn_tuned)
tuned_latency    = measure_latency(lr_adasyn_tuned, single_sample)
tuned_metrics["latency_ms"] = tuned_latency

print(f"\nBest Params   : {rs_cv.best_params_}")
print(f"Before Tuning : Recall={metrics_df[metrics_df['Variant']=='ADASYN'][metrics_df['Model']=='Logistic Regression']['Recall'].values[0]}  ROC-AUC={metrics_df[metrics_df['Variant']=='ADASYN'][metrics_df['Model']=='Logistic Regression']['ROC-AUC'].values[0]}")
print(f"After  Tuning : Recall={tuned_metrics['recall']}  ROC-AUC={tuned_metrics['roc_auc']}")
print(f"Latency       : {tuned_latency} ms")
print("\nClassification Report (LR ADASYN Tuned):")
print(tuned_metrics["report"])

all_metrics.append(tuned_metrics)
latency_results["Logistic Regression_ADASYN_Tuned"] = tuned_latency

# ─────────────────────────────────────────────
#  DECIDE: USE TUNED OR BASE LR ADASYN?
# ─────────────────────────────────────────────
base_lr_recall  = float(metrics_df[
    (metrics_df["Model"] == "Logistic Regression") &
    (metrics_df["Variant"] == "ADASYN")
]["Recall"].values[0])

tuned_lr_recall = tuned_metrics["recall"]

print("\n" + "="*65)
if tuned_lr_recall >= base_lr_recall:
    final_model   = lr_adasyn_tuned
    final_metrics = tuned_metrics
    final_variant = "ADASYN + Tuned"
    print(f"   Tuned model is BETTER  ({tuned_lr_recall} >= {base_lr_recall})")
    print(f"   Deploying LR ADASYN Tuned")
else:
    final_model   = trained_models["Logistic Regression_ADASYN"]
    final_metrics = [m for m in all_metrics
                     if m["model"] == "Logistic Regression"
                     and m["variant"] == "ADASYN"][0]
    final_variant = "ADASYN"
    print(f"   Base model is BETTER  ({base_lr_recall} >= {tuned_lr_recall})")
    print(f"   Deploying LR ADASYN Base")
print("="*65)

# ─────────────────────────────────────────────
#  SAVE BEST MODEL + ALL UPDATED METRICS
# ─────────────────────────────────────────────
joblib.dump(final_model, "models/best_model.joblib")
joblib.dump(all_metrics,  "models/all_metrics.joblib")
joblib.dump(latency_results, "models/latency_results.joblib")

cm = final_metrics["confusion_matrix"]

joblib.dump({
    "name":            "Logistic Regression",
    "variant":         final_variant,
    "params":          rs_cv.best_params_,
    "metrics":         final_metrics,
    "latency_results": latency_results,
    "reason": (
        f"Selected as the final deployed model because the business objective "
        f"is to minimize missed subscribers (False Negatives / Type 2 Error). "
        f"LR ADASYN ({final_variant}) achieves Recall of {final_metrics['recall']} "
        f"— missing only {cm[1][0]} out of {cm[1][0]+cm[1][1]} actual subscribers. "
        f"Prediction latency is {final_metrics['latency_ms']} ms per sample — "
        f"significantly faster than tree-based models, making it suitable for "
        f"real-time deployment at scale."
    ),
}, "models/best_model_info.joblib")

print(f"\n✅ Saved → models/best_model.joblib")
print(f"✅ Saved → models/best_model_info.joblib")
print(f"✅ Saved → models/all_metrics.joblib")
print(f"✅ Saved → models/latency_results.joblib")

print(f"\n{'='*65}")
print(f"   FINAL DEPLOYED MODEL SUMMARY")
print(f"{'='*65}")
print(f"   Model    : Logistic Regression  ({final_variant})")
print(f"   Accuracy : {final_metrics['accuracy']}")
print(f"   Recall   : {final_metrics['recall']}  ← business priority")
print(f"   F1 Score : {final_metrics['f1_score']}")
print(f"   ROC-AUC  : {final_metrics['roc_auc']}")
print(f"   Latency  : {final_metrics['latency_ms']} ms per prediction")
print(f"   Missed   : {cm[1][0]} out of {cm[1][0]+cm[1][1]} subscribers")
print(f"{'='*65}")
print(f"\n✅ models.py complete — run:  streamlit run app.py")