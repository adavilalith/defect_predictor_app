import pandas as pd
import os
import re

# --- Data Cleaning Helpers ---

def remove_duplicate_columns(df):
    """
    Removes duplicate columns that have different names but identical values.
    Returns a tuple: (cleaned DataFrame, duplicate_info).
    """
    duplicate_info = []
    unique_cols = []
    # Convert DataFrame to dictionary for easier comparison while handling NaNs
    df_compare = df.fillna('__NAN__')
    seen_values = {}

    for col in df_compare.columns:
        # Use a hashable representation of the column data
        col_data = tuple(df_compare[col].tolist())
        
        if col_data in seen_values:
            seen_values[col_data].append(col)
        else:
            seen_values[col_data] = [col]

    for cols in seen_values.values():
        if len(cols) > 1:
            kept_col = cols[0]
            removed_cols = cols[1:]
            duplicate_info.append((kept_col, removed_cols))
        else:
            kept_col = cols[0]

        unique_cols.append(kept_col)

    return df[unique_cols], duplicate_info

def remove_constant_columns(df):
    """
    Removes columns where all non-null values are identical (or only one unique non-null value exists).
    Returns a tuple: (cleaned DataFrame, list of removed columns).
    """
    removed_constants = []
    cols_to_keep = []

    for col in df.columns:
        # Count unique non-null values in the column
        unique_vals = df[col].nunique(dropna=True)
        if unique_vals <= 1:
            removed_constants.append(col)
        else:
            cols_to_keep.append(col)

    return df[cols_to_keep], removed_constants

def clean_dataframe(df):
    """
    Cleans the DataFrame by:
    1. Removing constant columns.
    2. Removing duplicate columns.
    3. Filling NaN values in numeric columns with their median.
    
    (Note: Steps 3 & 4 from original request are commented out to reflect your provided code.)
    """
    
    # 1. Remove constant columns
    df, removed_constants = remove_constant_columns(df)

    # 2. Remove duplicate columns
    df, duplicate_info = remove_duplicate_columns(df)

    # 3. Fill NaN values in numeric columns with median
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        median_value = df[col].median()
        if pd.notna(median_value):
            df[col] = df[col].fillna(median_value)

    return df

# --- Feature Engineering Helpers ---

def remove_params_simple(function_name):
    """Removes function parameters and simplifies namespaces."""
    # Remove anything inside parentheses and strip any extra spaces
    result = re.sub(r'\(.*\)', '', function_name).strip()

    if not result:
        return 'unknownfunction'

    # Handle C++ namespaces (only keep the last two parts)
    if '::' in result:
        parts = result.split('::')
        if len(parts) > 2:
            return '::'.join(parts[-2:])
        elif len(parts) == 2:
            return result
            
    return result

def extract_filename(path):
    """Extracts the filename from a full path."""
    filename = os.path.basename(path).strip()
    return filename if filename else 'unknownfile'

def preprocess_clang(df):
    """
    Applies feature engineering (name simplification, file extraction) and data cleaning.
    Assumes metrics columns are 'Function' and 'Location'.
    """
    
    # Apply transformations for consistency
    if 'Location' in df.columns:
        df['Location'] = df['Location'].apply(extract_filename)
        df = df.rename(columns={'Location': 'File_Name'})

    if 'Function' in df.columns:
        df['Function'] = df['Function'].apply(remove_params_simple)
        
    # Drop signature column if present
    if 'fSignature' in df.columns:
        df = df.drop(columns=['fSignature'])
        
    # Apply global cleaning (removes duplicates, constants, fills NaNs)
    df = clean_dataframe(df)
    
    return df

# --- Core Labeling Logic (Updated to use set-based matching and preprocessing) ---

def add_bug_label(df_metrics: pd.DataFrame, df_bugs: pd.DataFrame, bug_function_name_col: str) -> pd.DataFrame:
    """
    1. Preprocesses the metrics DataFrame.
    2. Labels metrics data by checking if the function exists in the bug report using efficient set comparison.

    Args:
        df_metrics (pd.DataFrame): DataFrame containing extracted metrics.
                                   Must have a 'Function' column after preprocessing.
        df_bugs (pd.DataFrame): DataFrame containing bug reports.
                                Must contain the column specified by bug_function_name_col.
        bug_function_name_col (str): The column name in df_bugs that lists buggy functions.

    Returns:
        pd.DataFrame: df_metrics augmented with a 'Bug' column (1 or 0).
    """

    # 1. Preprocess the metrics file (transforms, cleans, and renames columns)
    df_metrics = preprocess_clang(df_metrics)
    
    # Check for mandatory columns after preprocessing
    if 'Function' not in df_metrics.columns:
        raise ValueError("Metrics DataFrame must contain a 'Function' column after preprocessing.")
        
    # Check if the bug function column exists in the bug report dataframe
    if bug_function_name_col and bug_function_name_col not in df_bugs.columns:
        raise ValueError(f"Bug report CSV must contain the specified column: '{bug_function_name_col}'.")
    
    # --- Start Set-Based Labeling (Same as simple logic, but handles case-insensitivity explicitly) ---
    
    if not bug_function_name_col:
        # If no column is specified, label everything as non-buggy
        df_metrics['Bug'] = 0
        return df_metrics

    # 2. Prepare the list of buggy functions from the bug report
    # Normalize bug list: remove params (using same logic) and convert to lowercase for robust matching
    df_bugs['Normalized_Bug_Function'] = df_bugs[bug_function_name_col].astype(str).apply(remove_params_simple).str.lower()
    
    # Create a set of unique buggy function names for fast lookup
    buggy_functions_set = set(df_bugs['Normalized_Bug_Function'].dropna().unique())
    
    # 3. Normalize metrics function names for lookup
    df_metrics['Normalized_Metric_Function'] = df_metrics['Function'].astype(str).str.lower()
    
    # 4. Add 'Bug' column, default to 0 (False)
    df_metrics['Bug'] = 0
    
    # 5. Apply the label based on function name existence in the set
    is_buggy_mask = df_metrics['Normalized_Metric_Function'].isin(buggy_functions_set)
    
    df_metrics.loc[is_buggy_mask, 'Bug'] = 1
    
    # Clean up temporary columns used for matching
    df_metrics = df_metrics.drop(columns=['Normalized_Metric_Function'])
    
    return df_metrics