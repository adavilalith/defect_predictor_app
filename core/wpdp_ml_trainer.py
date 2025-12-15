# core/wpdp_ml_trainer.py

import pandas as pd
import numpy as np
import os
import time
import json
import cloudpickle

# ML Library Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.base import BaseEstimator 

# --- NEW IMPORTS for Graphical Confusion Matrix ---
import matplotlib.pyplot as plt 
from sklearn.metrics import ConfusionMatrixDisplay
# ---------------------------------------------------

# Assuming model_configs.py is available in the 'core' directory
from .model_configs import MODEL_CONFIGS 

# Define the expected binary class labels explicitly for consistent metrics/CM
LABELS = [0, 1] 

# --- Helper Function: Model Saving and Logging (UPDATED TO INCLUDE FEATURES) ---
def save_model_and_log(trained_model, scaler, final_features, results, config):
    """
    Saves the trained model, scaler, final feature list, and a log file (JSON) 
    to the output directory.
    """
    if not config['save_model']:
        return None, None # Return None if not saving

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name_clean = config['model'].replace(' ', '_').replace('/', '_')
    run_dir = os.path.join(config['output_dir'], f"{model_name_clean}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 1. Save Model 
    model_path = os.path.join(run_dir, "model.pkl")
    try:
        with open(model_path, 'wb') as f:
            cloudpickle.dump(trained_model, f)
    except Exception as e:
        raise IOError(f"Failed to save model file to {model_path}. Check file permissions/path. Error: {e}")

    # 2. Save Scaler (Only if normalization was applied)
    if config['normalize'] and scaler is not None:
        scaler_path = os.path.join(run_dir, "scaler.pkl")
        try:
            with open(scaler_path, 'wb') as f:
                cloudpickle.dump(scaler, f)
        except Exception as e:
            raise IOError(f"Failed to save scaler file to {scaler_path}. Error: {e}")

    # 3. Save Final Features List (CRITICAL FOR INFERENCE)
    features_path = os.path.join(run_dir, "features.json")
    try:
        with open(features_path, 'w') as f:
            json.dump(final_features, f, indent=4)
    except Exception as e:
        raise IOError(f"Failed to save feature list file to {features_path}. Error: {e}")

    # 4. Save Log (configuration and results)
    # Filter out non-serializable objects (like the CM plot image path if it's a string error)
    log_results = {k: v for k, v in results.items() if k != 'CM Plot Path' or isinstance(v, str) and 'Error' not in v}

    log_data = {
        "timestamp": timestamp,
        "model_name": config['model'],
        "configuration": config,
        "results": log_results
    }
    log_path = os.path.join(run_dir, "run_log.json")
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)
        
    return run_dir, model_path # Return the run directory and model path

# --- Helper Function: Feature Selection (No change) ---
def apply_feature_selection(X_train, X_test, y_train, config):
    """Applies the configured feature selection method."""
    
    if not config['fs_apply']:
        return X_train, X_test, X_train.columns.tolist()

    method = config['fs_method']
    
    if method == "CSV Filter":
        if not os.path.isfile(config['fs_csv_path']):
            raise FileNotFoundError(f"Feature list CSV not found at: {config['fs_csv_path']}")
        
        feature_df = pd.read_csv(config['fs_csv_path'], header=None)
        selected_features = feature_df.iloc[1:,0].tolist()
        selected_features = [f.upper() for f in selected_features]

        final_features = [f for f in selected_features if f in X_train.columns]

        if not final_features:
            raise ValueError("CSV Filter resulted in zero matching features.")

        X_train = X_train[final_features]
        X_test = X_test[final_features]
        return X_train, X_test, final_features

    elif method == "SelectKBest (Chi2)":
        k = config['fs_k']
        selector = SelectKBest(score_func=chi2, k=min(k, X_train.shape[1]))
        X_train_new = selector.fit_transform(X_train, y_train)
        X_test_new = selector.transform(X_test)
        
        mask = selector.get_support()
        final_features = X_train.columns[mask].tolist()
        
        return pd.DataFrame(X_train_new, columns=final_features), pd.DataFrame(X_test_new, columns=final_features), final_features

    elif method == "RFE (Recursive Feature Elimination)":
        k = config['fs_k']
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        
        estimator = LogisticRegression(solver='liblinear', random_state=42)
        rfe = RFE(estimator, n_features_to_select=min(k, X_train.shape[1]), step=1)
        
        X_train_new = rfe.fit_transform(X_train, y_train)
        X_test_new = rfe.transform(X_test)
        
        mask = rfe.support_
        final_features = X_train.columns[mask].tolist()

        return pd.DataFrame(X_train_new, columns=final_features), pd.DataFrame(X_test_new, columns=final_features), final_features
    
    else:
        # Should be unreachable if UI enforces valid method, but included for robustness.
        return X_train, X_test, X_train.columns.tolist()


# --- Main Logic Function (UPDATED to pass FINAL_FEATURES to saving function) ---
def run_wpdp_ml_experiment(config: dict, progress_callback=None) -> dict:
    """
    Executes the full ML pipeline based on user configuration.
    """
    if progress_callback is None:
        progress_callback = lambda p: None

    progress_callback({'percent': 5, 'message': 'Starting experiment and loading data...'})
    
    # 1. Data Loading and Splitting 
    try:
        df = pd.read_csv(config['data_path'])
    except Exception as e:
        raise ValueError(f"Error loading CSV data: {e}")

    ignore_cols = ['Bug', 'function_name', 'filepath', 'commit_hash']
    if 'Bug' not in df.columns:
        raise ValueError("Input CSV must contain a 'Bug' column for training.")
        
    X = df.drop(columns=[col for col in ignore_cols if col in df.columns], errors='ignore')
    y = df['Bug']
    
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    X = X[numeric_cols]

    test_size = 1 - config['split_ratio']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    progress_callback({'percent': 15, 'message': f'Data split: {config["split_ratio"]*100:.0f}% train / {test_size*100:.0f}% test.'})

    # Variable to hold the fitted scaler (MinMaxScaler), used for saving.
    scaler = None 

    # 2. Preprocessing (Normalization)
    if config['normalize']:
        progress_callback({'percent': 20, 'message': 'Applying MinMaxScaler...'})
        scaler = MinMaxScaler() # Initialize scaler
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # 3. Feature Selection
    progress_callback({'percent': 30, 'message': 'Applying Feature Selection...'})
    # --- GET FINAL_FEATURES HERE ---
    X_train, X_test, final_features = apply_feature_selection(X_train, X_test, y_train, config)
    progress_callback({'percent': 40, 'message': f'Feature Selection complete. {len(final_features)} features remaining.'})

    # 4. Oversampling (SMOTE)
    if config['smote']:
        progress_callback({'percent': 45, 'message': 'Applying SMOTE oversampling...'})
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
        
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train_np, y_train_np)
        
        X_train = pd.DataFrame(X_train_res, columns=final_features)
        y_train = pd.Series(y_train_res)
        
    progress_callback({'percent': 50, 'message': 'Preprocessing pipeline finalized.'})
    
    # 5. Model Training
    progress_callback({'percent': 60, 'message': f"Training {config['model']} model..."})
    
    model_info = MODEL_CONFIGS.get(config['model'])
    if not model_info:
        raise ValueError(f"Model '{config['model']}' not found in model configurations.")

    model_class = model_info['class']
    ModelClass = model_class
    
    try:
        model = ModelClass(**config['hyperparams'], random_state=42)
    except Exception as e:
        raise ValueError(f"Failed to initialize model {config['model']} with provided parameters: {e}")

    model.fit(X_train, y_train)
    progress_callback({'percent': 75, 'message': 'Model training complete.'})

    # 6. Evaluation
    progress_callback({'percent': 80, 'message': 'Evaluating model performance...'})
    y_pred = model.predict(X_test)

    # --- Metrics Calculation ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=LABELS, zero_division=0)
    recall = recall_score(y_test, y_pred, labels=LABELS, zero_division=0)
    f1 = f1_score(y_test, y_pred, labels=LABELS, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    report_text = classification_report(y_test, y_pred, labels=LABELS, output_dict=False, zero_division=0)

    # Temporary path handling for CM Plot
    cm_fig_path = "N/A"
    
    # Determine the directory where the plot will be temporarily saved or finally saved
    if config.get('save_model') and config.get('output_dir'):
        target_dir = config.get('output_dir')
        cm_base_filename = "confusion_matrix.png"
    else:
        # Use a temporary directory if not saving the model
        target_dir = os.path.join("/tmp", "wpdp_ml_temp")
        cm_base_filename = f"cm_plot_{time.time()}.png"
        
    try:
        os.makedirs(target_dir, exist_ok=True)
        cm_fig_path = os.path.join(target_dir, cm_base_filename)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Bug (0)", "Bug (1)"])
        disp.plot(cmap=plt.cm.Blues) 
        plt.title(f"Confusion Matrix ({config['model']})")
        plt.savefig(cm_fig_path)
        plt.close() 
    
    except Exception as e:
        cm_fig_path = f"Error generating CM plot: {type(e).__name__}: {str(e)}"
        print(f"CM Plot Error: {cm_fig_path}")

    final_results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': cm.tolist(),
        'Classification Report Text': report_text,
        'Feature Count': len(final_features),
        'CM Plot Path': cm_fig_path, 
    }

    progress_callback({'percent': 90, 'message': 'Evaluation finished.'})

    # 7. Saving (UPDATED TO PASS SCALER AND FINAL_FEATURES)
    if config['save_model']:
        progress_callback({'percent': 95, 'message': 'Saving model and logs...'})
        # --- PASS THE SCALER AND FINAL_FEATURES HERE ---
        run_dir, model_path = save_model_and_log(model, scaler, final_features, final_results, config) 
        
        # FINAL STEP: Ensure the CM Plot is moved/renamed correctly into the final run directory
        final_cm_path = os.path.join(run_dir, "confusion_matrix.png")
        if os.path.isfile(cm_fig_path) and cm_fig_path != final_cm_path:
            # If the plot was temporarily saved outside the final run_dir, move it in.
            os.rename(cm_fig_path, final_cm_path)
            final_results['CM Plot Path'] = final_cm_path
        
        final_results['Save Path'] = run_dir
        
    progress_callback({'percent': 100, 'message': 'Experiment completed successfully.'})

    return final_results