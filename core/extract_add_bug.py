import pandas as pd
import os
from core.labeling_logic import add_bug_label 
from core.metrics_extractor import MetricsExtractor



def extract_metrics_and_add_bug_label(source_folder: str, bug_report_csv: str, 
                                      output_csv: str, bug_function_name_col: str, 
                                      progress_callback=None) -> pd.DataFrame:
    """
    Extract metrics from source code and add bug labels from bug report.
    
    Args:
        source_folder (str): Path to source code folder.
        bug_report_csv (str): Path to bug report CSV file.
        output_csv (str): Path to save the output CSV.
        bug_function_name_col (str): The column in df_bugs containing buggy function names.
        progress_callback (callable): Function to report progress (0-100).
        
    Returns:
        pandas.DataFrame: The processed DataFrame with metrics and bug labels.
    """
    
    if progress_callback is None:
        progress_callback = lambda p: None

    try:
        # 1. Extract Metrics
        progress_callback(5.0)
        extractor = MetricsExtractor()
        
        def extraction_progress_reporter(percent):
             # Scale extraction progress (0-50% overall)
            progress_callback(5 + percent * 0.45) 
        
        # Call the extractor's main method to get the metrics DataFrame
        df_metrics = extractor.process_folder(
            source_folder,
            None, # Pass None as output path to keep result in memory for immediate labeling
            progress_callback=extraction_progress_reporter
        )
        
        progress_callback(50.0)

        # 2. Read Bug Report
        if not os.path.exists(bug_report_csv):
             raise FileNotFoundError(f"Bug report CSV not found: {bug_report_csv}")
             
        df_bugs = pd.read_csv(bug_report_csv)
        progress_callback(60.0)
        
        # 3. Add Bug Labels
        # Passes the extracted metrics, the bug report data, and the target column name
        result_df = add_bug_label(df_metrics, df_bugs, bug_function_name_col)
        
        progress_callback(85.0)
        
        # 4. Save the result to the specified output path
        if output_csv:
            result_df.to_csv(output_csv, index=False)
        
        progress_callback(100.0)
        
        return result_df
        
    except Exception as e:
        # Re-raise the exception to be handled by the caller (the Worker thread)
        raise Exception(f"Data Preparation Error: {str(e)}")