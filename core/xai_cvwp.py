#core/xai_cvwp.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import shap

def cvwp_shap_multiple_csv(train_csvs, test_csvs, model_class, model_params=None, shap_threshold=0.75, smote_random_state=42):
    """
    Run SHAP-based feature selection and evaluation for multiple train/test CSVs merged.
    Always classification, no normalization, SMOTE with sampling_strategy=0.5.

    Returns:
        metrics_df : pd.DataFrame (baseline vs SHAP metrics + k_cum)
        features_df : pd.DataFrame (top-k features ranked)
        shap_df : pd.DataFrame (all features ranked by SHAP importance)
    """
    # Merge all train and test CSVs
    train_df = pd.concat([pd.read_csv(f) for f in train_csvs], ignore_index=True)
    test_df  = pd.concat([pd.read_csv(f) for f in test_csvs], ignore_index=True)


    # Lowercase columns
    train_df.columns = train_df.columns.str.lower()
    test_df.columns  = test_df.columns.str.lower()

    # Drop unwanted columns
    for col in ["location", "function"]:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])

    target_col = "bug"
    feature_cols = [c for c in train_df.columns if c != target_col]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test  = test_df[feature_cols]
    y_test  = test_df[target_col]

    # Apply SMOTE
    smote = SMOTE(sampling_strategy=0.5, random_state=smote_random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    if model_params is None:
        model_params = {}

    # ---- Baseline training ----
    model = model_class(**model_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "baseline_accuracy": accuracy_score(y_test, y_pred),
        "baseline_precision": precision_score(y_test, y_pred, zero_division=0),
        "baseline_recall": recall_score(y_test, y_pred, zero_division=0),
        "baseline_f1": f1_score(y_test, y_pred, zero_division=0)
    }

    # ---- SHAP feature selection ----
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Binary classification
    shap_array = shap_values.values[:,:,1] if shap_values.values.ndim==3 else shap_values.values
    mean_abs_shap = np.abs(shap_array).mean(axis=0)

    shap_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap
    }).sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)

    # Top-k features
    cumulative = np.cumsum(shap_df['mean_abs_shap'].values) / np.sum(shap_df['mean_abs_shap'].values)
    k_cum = np.argmax(cumulative >= shap_threshold) + 1
    top_features = shap_df['feature'].iloc[:k_cum].tolist()

    # Train on top-k features
    X_train_top = X_train[top_features]
    X_test_top  = X_test[top_features]
    model.fit(X_train_top, y_train)
    y_pred_top = model.predict(X_test_top)

    # SHAP metrics
    metrics.update({
        "shap_accuracy": accuracy_score(y_test, y_pred_top),
        "shap_precision": precision_score(y_test, y_pred_top, zero_division=0),
        "shap_recall": recall_score(y_test, y_pred_top, zero_division=0),
        "shap_f1": f1_score(y_test, y_pred_top, zero_division=0),
        "k_cum": k_cum
    })

    metrics_df = pd.DataFrame([metrics])

    # Features ranked
    features_df = pd.DataFrame({"feature": top_features})
    features_df["rank"] = np.arange(1, len(features_df)+1)

    return metrics_df, features_df, shap_df

