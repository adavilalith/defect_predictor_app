
import pandas as pd
import os
from core.labeling_logic import add_bug_label


def process_existing_metrics_and_add_bug(
    metrics_csv_path: str, 
    bug_report_csv: str, 
    output_csv: str, 
    bug_function_name_col: str, 
    progress_callback=None
) -> pd.DataFrame:
    """
    Loads an existing DataFrame of code metrics, applies necessary 
    preprocessing and cleaning, and adds a bug label based on a bug report CSV.

    Args:
        metrics_csv_path (str): Path to the existing raw metrics CSV file.
        bug_report_csv (str): Path to bug report CSV file (containing buggy functions).
        output_csv (str): Path to save the final labeled output CSV.
        bug_function_name_col (str): The column in df_bugs containing buggy function names.
        progress_callback (callable): Function to report progress (0-100).
        
    Returns:
        pandas.DataFrame: The processed DataFrame with metrics and bug labels.
    """
    if progress_callback is None:
        progress_callback = lambda p: None

    # --- 1. Load Existing Metrics (0% - 20%) ---
    progress_callback(5.0)
    
    if not os.path.exists(metrics_csv_path):
        raise FileNotFoundError(f"Existing metrics CSV not found: {metrics_csv_path}")
    
    df_metrics = pd.read_csv(metrics_csv_path)
    progress_callback(20.0)

    # --- 2. Load Bug Report (20% - 40%) ---
    if not os.path.exists(bug_report_csv):
        raise FileNotFoundError(f"Bug report CSV not found: {bug_report_csv}")
        
    df_bugs = pd.read_csv(bug_report_csv)
    progress_callback(40.0)

    # --- 3. Preprocessing, Cleaning, and Labeling (40% - 80%) ---
    # The 'add_bug_label' function handles all steps V (Preprocessing) and VI (Labeling)
    # including function name normalization and set-based lookup.
    
    try:
        # Note: We assume 'add_bug_label' is a universal function that takes 
        # raw metrics (post-libclang structure) and the bug report, 
        # and internally applies preprocessing before labeling.
        result_df = add_bug_label(df_metrics, df_bugs, bug_function_name_col)
        progress_callback(80.0)
        
    except Exception as e:
        # Catch errors specific to the labeling/preprocessing step (e.g., missing columns)
        raise Exception(f"Labeling/Preprocessing Error: {str(e)}")

    # --- 4. Save the result (80% - 100%) ---
    if output_csv:
        result_df.to_csv(output_csv, index=False)
        
    progress_callback(100.0)
    
    return result_df