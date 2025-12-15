#core/xai_wvwp.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import shap

def wvwp_shap_single_csv(
    csv_path,
    model_class,
    model_params=None,
    feature_cols=None,
    normalize=False,
    apply_smote=True,
    sampling_strategy=0.5,
    smote_random_state=42,
    train_test_split_ratio=0.8,
    shap_threshold=0.75,
    random_state=42
):
    """
    Train and evaluate a classifier with SHAP-based feature selection from a single CSV.
    Always:
      - Task type: classification
      - Target column: 'bug'
      - Drop 'location' and 'function'
      - Columns converted to lowercase
    """

    # Load data
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    df = df.drop(columns=[c for c in ["location", "function"] if c in df.columns])

    target_col = "bug"
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_test_split_ratio, random_state=random_state
    )

    # Apply SMOTE
    if apply_smote:
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=smote_random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    if model_params is None:
        model_params = {}

    # ---- Baseline training ----
    model = model_class(**model_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Baseline metrics
    metrics = {
        "baseline_accuracy": accuracy_score(y_test, y_pred),
        "baseline_precision": precision_score(y_test, y_pred, zero_division=0),
        "baseline_recall": recall_score(y_test, y_pred, zero_division=0),
        "baseline_f1": f1_score(y_test, y_pred, zero_division=0),
    }

    # ---- SHAP feature selection ----
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    shap_array = shap_values.values[:,:,1] if shap_values.values.ndim==3 else shap_values.values
    mean_abs_shap = np.abs(shap_array).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    # Top-k features (cumulative threshold)
    cumulative = np.cumsum(shap_df["mean_abs_shap"].values) / shap_df["mean_abs_shap"].sum()
    k_cum = np.argmax(cumulative >= shap_threshold) + 1
    top_features = shap_df["feature"].iloc[:k_cum].tolist()

    # Train on top-k
    X_train_top, X_test_top = X_train[top_features], X_test[top_features]
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
    features_df = pd.DataFrame({"feature": top_features, "rank": range(1, len(top_features)+1)})

    return metrics_df, features_df, shap_df

