# Enhanced default_model_tab.py with QSplitter and Robust Analytics/Prediction
import os
import shutil
import tempfile
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from functools import partial

# PyQt5 Imports
from PyQt5.QtWidgets import (
    QWidget, QFormLayout, QPushButton, QLabel, QCheckBox, 
    QLineEdit, QFileDialog, QMessageBox, QVBoxLayout, QScrollArea,
    QSizePolicy, QGroupBox, QTableWidget, QTableWidgetItem, QTabWidget,
    QApplication, QHBoxLayout, QProgressBar, QSplitter
)
from PyQt5.QtCore import Qt

# Scikit-learn Imports
from sklearn.preprocessing import StandardScaler

# Ensure this import exists in your project structure
# from ui.components.csv_analytics_dialog import CSVAnalyticsDialog 
# Assuming it exists, if not, you'll need to create it or remove the import.
try:
    from ui.components.csv_analytics_dialog import CSVAnalyticsDialog
except ImportError:
    # Define a dummy class if the real one is not available to prevent crashes
    logging.warning("CSVAnalyticsDialog not found. Detailed analytics disabled.")
    class CSVAnalyticsDialog(object):
        def __init__(self, *args, **kwargs): pass
        def show(self): 
            QMessageBox.information(None, "Information", "Detailed analytics dialog not available.")


# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DefaultModelTab(QWidget):
    """
    UI and logic for running default bug prediction models on a CSV dataset.
    Features include: CSV data loading, analytics preview, data normalization, 
    prediction execution with multiple models, and results download.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.csv_file: str = ""
        self.current_csv_path: str = ""
        self.models: dict = {}
        self.prediction_results: dict[str, pd.DataFrame] = {}
        self.temp_csv_paths: dict[str, str] = {}
        
        # Mapping for model keys to display names and table indices
        self.model_info = {
            'p1_local': {'name': 'P1 Local Model', 'index': 0},
            'p2_local': {'name': 'P2 Local Model', 'index': 1},
            'p3_local': {'name': 'P3 Local Model', 'index': 2},
            'global':   {'name': 'Global Model',   'index': 3},
        }

        self.setup_ui()
        self.load_models()
        
    def setup_ui(self):
        """Setup the main UI with QSplitter for resizable sections."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5) # Added minor margin

        # QSplitter for control area (top) and preview area (bottom)
        splitter = QSplitter(Qt.Orientation.Vertical)

        # 1. Input Controls Section (Top - scrollable)
        input_scroll_area = QScrollArea()
        input_scroll_area.setWidgetResizable(True)
        input_container = QWidget()
        self.input_layout = QVBoxLayout(input_container)
        self.input_layout.setSpacing(15)
        self.input_layout.setContentsMargins(10, 10, 10, 10)
        
        self._create_input_section()
        
        input_scroll_area.setWidget(input_container)
        splitter.addWidget(input_scroll_area)

        # 2. Prediction Results Preview Section (Bottom - resizable)
        self._create_preview_section()
        splitter.addWidget(self.preview_box)

        # Set initial sizes for the splitter (e.g., 60% for controls, 40% for results)
        splitter.setSizes([600, 400])

        main_layout.addWidget(splitter)

        # Progress bar (outside splitter)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

    def _create_input_section(self):
        """Creates and adds the file selection and configuration widgets."""
        input_widget = QWidget()
        form_layout = QFormLayout(input_widget)
        form_layout.setSpacing(20)
        
        # --- 1. File Selection Section ---
        file_section_label = QLabel("📥 Input File:")
        file_section_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #1E88E5;")
        form_layout.addRow(file_section_label)
        
        # CSV Input Field with Browse Button
        csv_input_layout = QHBoxLayout()
        self.csv_lineedit = QLineEdit()
        self.csv_lineedit.setPlaceholderText("No file selected - e.g., metrics_data.csv")
        self.csv_lineedit.setReadOnly(True)
        browse_btn = QPushButton("Browse")
        browse_btn.setToolTip("Select the CSV file containing function metrics.")
        browse_btn.clicked.connect(self.browse_file)
        csv_input_layout.addWidget(self.csv_lineedit)
        csv_input_layout.addWidget(browse_btn)
        form_layout.addRow("Select CSV File:", csv_input_layout)
        
        # CSV Info Group (Hidden initially)
        self.csv_info_group = QGroupBox("📊 Dataset Information")
        self.csv_info_group.setVisible(False)
        info_layout = QFormLayout(self.csv_info_group)
        
        self.file_name_label = QLabel()
        self.file_size_label = QLabel()
        self.rows_label = QLabel()
        self.columns_label = QLabel()
        self.missing_data_label = QLabel()
        
        info_layout.addRow("File Name:", self.file_name_label)
        info_layout.addRow("File Size:", self.file_size_label)
        info_layout.addRow("Rows:", self.rows_label)
        info_layout.addRow("Columns:", self.columns_label)
        info_layout.addRow("Missing Data:", self.missing_data_label)
        
        # Analytics button
        self.analytics_btn = QPushButton("View Detailed Analytics")
        self.analytics_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; 
                color: white; 
                padding: 8px; 
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.analytics_btn.clicked.connect(self.show_analytics)
        self.analytics_btn.setVisible(False)
        
        info_layout.addRow(self.analytics_btn)
        form_layout.addRow(self.csv_info_group)
        
        # Required Info
        info_label = QLabel("• **Required:** CSV file with function metrics (e.g., from 'Extract Metrics' feature).\n"
                            "• The file must contain the necessary feature columns for model prediction.")
        info_label.setStyleSheet("color: #666; font-size: 12px; margin: 5px 0px; background-color: transparent;")
        info_label.setWordWrap(True)
        form_layout.addRow(info_label)
        
        # --- 2. Configuration Section ---
        config_section_label = QLabel("⚙️ Configuration:")
        config_section_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #1E88E5;")
        form_layout.addRow(config_section_label)
        
        # Normalize option
        self.normalize_cb = QCheckBox("Normalize Data (Standard Scaling)")
        self.normalize_cb.setToolTip("Apply StandardScaler to feature columns before prediction.")
        form_layout.addRow(self.normalize_cb)
        
        normalize_info_label = QLabel("• Standardize feature values (mean=0, variance=1) to match training data.\n")
        normalize_info_label.setStyleSheet("color: #666; font-size: 12px; margin: 5px 0px; background-color: transparent;")
        normalize_info_label.setWordWrap(True)
        form_layout.addRow(normalize_info_label)
        
        # Run button
        self.run_btn = QPushButton("🚀 Run Default Prediction")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                padding: 12px; 
                font-weight: bold; 
                font-size: 14px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
        """)
        self.run_btn.clicked.connect(self.run_default_prediction)
        form_layout.addRow(self.run_btn)
        
        self.input_layout.addWidget(input_widget)
        # Add a spacer to push elements to the top
        self.input_layout.addStretch(1)

    def _create_preview_section(self):
        """Creates the prediction results QTabWidget and download buttons."""
        self.preview_box = QGroupBox("Results Preview and Download")
        self.preview_box.setVisible(False)
        preview_layout = QVBoxLayout(self.preview_box)

        # Create tab widget for the prediction results
        self.preview_tabs = QTabWidget()
        
        for key, info in self.model_info.items():
            model_name = info['name']
            table_idx = info['index']
            
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            
            # Create table for this model
            table = QTableWidget()
            table.setAlternatingRowColors(True)
            table.horizontalHeader().setStretchLastSection(True)
            table.setEditTriggers(QTableWidget.NoEditTriggers) # Make table read-only
            table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            table.setMinimumHeight(200)
            
            # Download button using functools.partial for passing arguments safely
            download_btn = QPushButton(f"💾 Download {model_name} Results CSV")
            download_btn.clicked.connect(partial(self.download_csv, key))
            
            tab_layout.addWidget(table)
            tab_layout.addWidget(download_btn)
            
            self.preview_tabs.addTab(tab, model_name)
            
            # Store reference to table dynamically
            setattr(self, f'table_{key}', table)

        preview_layout.addWidget(self.preview_tabs)

    def browse_file(self):
        """Open file dialog to select CSV file."""
        # Use QFileDialog.getOpenFileName for standard file selection
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", str(Path.home()), "CSV Files (*.csv)"
        )
        if file_path:
            self.csv_file = file_path
            self.csv_lineedit.setText(Path(file_path).name)
            self.load_csv_info(file_path)

    def load_csv_info(self, csv_path: str):
        """Load and display basic CSV information."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate progress
        QApplication.processEvents()
        
        try:
            full_df = pd.read_csv(csv_path)
            
            # Basic stats
            file_size_bytes = Path(csv_path).stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            num_rows = len(full_df)
            num_cols = len(full_df.columns)
            
            # Missing data calculation
            missing_count = full_df.isnull().sum().sum()
            total_cells = num_rows * num_cols
            missing_percent = (missing_count / total_cells) * 100 if total_cells > 0 else 0
            
            # Update labels
            self.file_name_label.setText(Path(csv_path).name)
            self.file_size_label.setText(f"{file_size_mb:,.2f} MB")
            self.rows_label.setText(f"{num_rows:,}")
            self.columns_label.setText(f"{num_cols}")
            
            if missing_count > 0:
                self.missing_data_label.setText(f"{missing_count:,.0f} ({missing_percent:.2f}%)")
                self.missing_data_label.setStyleSheet("color: #D32F2F; font-weight: bold;") # Reddish/Orange
            else:
                self.missing_data_label.setText("None")
                self.missing_data_label.setStyleSheet("color: #388E3C; font-weight: bold;") # Green
            
            # Show the info section and analytics button
            self.csv_info_group.setVisible(True)
            self.analytics_btn.setVisible(True)
            self.current_csv_path = csv_path
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error loading CSV file for info:\n{str(e)}")
            self.csv_info_group.setVisible(False)
            self.analytics_btn.setVisible(False)
            logging.error("Failed to load CSV info: %s", e)
        finally:
            self.progress_bar.setVisible(False)

    def show_analytics(self):
        """Show detailed analytics in a popup dialog using CSVAnalyticsDialog."""
        if hasattr(self, 'current_csv_path') and self.current_csv_path:
            # Check if CSVAnalyticsDialog is the dummy or the real one
            if CSVAnalyticsDialog.__name__ == 'CSVAnalyticsDialog' and CSVAnalyticsDialog is not object:
                analytics_dialog = CSVAnalyticsDialog(self.current_csv_path, self)
                analytics_dialog.exec_() # Use exec_() for modal dialog
            else:
                 self.show_basic_analytics()
        else:
            QMessageBox.warning(self, "Warning", "No CSV file selected.")

    def show_basic_analytics(self):
        """Fallback: Show basic pandas describe in a message box."""
        try:
            df = pd.read_csv(self.current_csv_path)
            
            stats_text = f"""
**Dataset Overview:**
• Shape: {df.shape[0]} rows × {df.shape[1]} columns
• Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

**Numeric Column Summary (First 10 columns):**
{df.select_dtypes(include=[np.number]).iloc[:, :10].describe().to_string()}
            """
            QMessageBox.information(self, "Basic CSV Analytics (Fallback)", stats_text)
        except Exception as e:
            QMessageBox.critical(self, "Analytics Error", f"Error generating basic analytics: {str(e)}")

    def load_models(self):
        """Load all pre-trained models from the 'core/models' directory."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        QApplication.processEvents()

        current_dir = Path(__file__).parent
        
        try:
            model_dir = current_dir.parent.parent / "core" / "models"
        except Exception:
            # Fallback for simpler structures or unexpected location
            model_dir = current_dir / "core" / "models" 
        
        # Sanity check: If model_dir still doesn't exist, try just 'core/models' relative to CWD,
        # but sticking to Path(__file__) is more reliable in complex projects.
        if not model_dir.is_dir():
            logging.warning("Primary model path not found: %s. Trying relative path.", model_dir)
            model_dir = Path("core") / "models" # Try relative to CWD as fallback 

        logging.info("Attempting to load models from: %s", model_dir.resolve())
        
        model_filenames = {
            'p1_local': 'p1_local_model.joblib',
            'p2_local': 'p2_local_model.joblib',
            'p3_local': 'p3_local_model.joblib',
            'global': 'global_model.joblib',
        }
        
        loaded_count = 0
        self.models.clear()
        
        try:
            for key, filename in model_filenames.items():
                model_path = model_dir / filename
                if model_path.exists():
                    self.models[key] = self._load_model_with_xgboost_fix(str(model_path))
                    logging.info("Successfully loaded model: %s from %s", key, model_path.name)
                    loaded_count += 1
                else:
                    logging.warning("Model file not found for %s: %s", key, model_path.resolve())
                    
            if loaded_count == 0:
                 QMessageBox.warning(self, "Model Warning", "No models were loaded. Check the 'core/models' directory and file names.")
            
        except Exception as e:
            logging.error("Critical error during model loading: %s", str(e))
            QMessageBox.critical(self, "Model Loading Error", 
                                 f"Failed to load pre-trained models: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            
    def _load_model_with_xgboost_fix(self, model_path: str):
        """
        Loads a joblib model, applying necessary fixes, especially for older 
        XGBoost versions or compatibility issues.
        """
        model = joblib.load(model_path)
        
        # XGBoost compatibility fix for use_label_encoder warning/error
        if hasattr(model, 'get_booster'):
            if hasattr(model, 'use_label_encoder'):
                try:
                    # Setting to False is the recommended way to suppress the warning
                    model.set_params(use_label_encoder=False)
                except Exception:
                    pass
        
        return model

    def _preprocess_data(self, df: pd.DataFrame, normalize: bool):
        """
        Preprocess the input data (DataFrame) to match training format.
        
        1. Remove identifying columns ('Function', 'Location').
        2. Separate target variable ('Bug') if present.
        3. Handle missing values (fill with 0, as is common for metric data).
        4. Normalize data using StandardScaler if requested.
        """
        # Columns to keep track of, but remove from feature set
        cols_to_remove = ['Function', 'Location']
        df_processed = df.drop(columns=cols_to_remove, errors='ignore')

        # Separate target column if it exists (for validation/comparison)
        y_true = None
        if 'Bug' in df_processed.columns:
            y_true = df_processed['Bug'].values
            df_processed = df_processed.drop(columns=['Bug'])
            logging.info("Target 'Bug' column found and separated for validation.")

        # Handle missing values: Fill NaN with 0
        df_processed = df_processed.fillna(0)
        logging.info("Missing values filled with 0.")

        # Normalize if requested
        if normalize:
            scaler = StandardScaler()
            # Ensure only numeric columns are scaled
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
                logging.info("Data normalized using StandardScaler on numeric columns.")
            else:
                logging.warning("Normalization requested but no numeric columns found.")

        return df_processed, y_true

    def _predict_with_model(self, model, X_test: pd.DataFrame, model_key: str):
        """
        Make predictions with a specific model, ensuring feature alignment.
        """
        required_features = X_test.columns.tolist() # Default to current columns

        try:
            # 1. Get features the model was trained on
            if hasattr(model, 'feature_names_in_'):
                required_features = model.feature_names_in_.tolist()
            elif hasattr(model, 'get_booster'):
                try:
                    booster_features = model.get_booster().feature_names
                    if booster_features:
                         required_features = booster_features
                except Exception:
                    pass

            # 2. Align features: Add missing features (fill with 0) and reorder
            X_test_aligned = X_test.copy()
            for feature in required_features:
                if feature not in X_test_aligned.columns:
                    X_test_aligned[feature] = 0
                    logging.debug("Added missing feature: %s for model %s", feature, model_key)
            
            # Select and reorder columns to match the trained model's feature order
            X_test_model = X_test_aligned[required_features]
            
            # 3. Make predictions
            predictions = model.predict(X_test_model)
            probabilities = None
            
            # Get probabilities if the model supports it
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(X_test_model)
                except Exception as e:
                    logging.warning("Failed to get probabilities for %s: %s", model_key, str(e))

            # Handle raw XGBoost predictions (if necessary)
            if predictions is None and hasattr(model, 'get_booster'):
                # This fallback handles cases where direct model.predict() fails
                import xgboost as xgb
                dmatrix = xgb.DMatrix(X_test_model)
                raw_predictions = model.get_booster().predict(dmatrix)
                
                if len(raw_predictions.shape) > 1:
                    predictions = np.argmax(raw_predictions, axis=1)
                    probabilities = raw_predictions
                else:
                    predictions = (raw_predictions > 0.5).astype(int)
                    probabilities = raw_predictions
            
            return predictions, probabilities

        except Exception as e:
            logging.error("Prediction failed for model %s: %s", model_key, str(e))
            return None, None

    def run_default_prediction(self):
        """Execute the default model prediction workflow."""
        if not self.csv_file:
            QMessageBox.warning(self, "Missing Input", "Please select a CSV file for prediction.")
            return
        
        if not self.models:
            QMessageBox.warning(self, "No Models", "No pre-trained models are loaded. Check log for loading errors.")
            return

        # Start UI feedback
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Processing...")
        QApplication.processEvents()
        
        try:
            # Load and preprocess data
            df = pd.read_csv(self.csv_file)
            original_df = df.copy().reset_index(drop=True) # Ensure consistent index for merging
            
            normalize = self.normalize_cb.isChecked()
            X_test, y_true = self._preprocess_data(df, normalize)
            
            # Clear previous results
            self.prediction_results.clear()
            self.temp_csv_paths.clear()
            successful_predictions = 0
            
            # Make predictions with each model
            for model_key, info in self.model_info.items():
                if model_key in self.models:
                    model = self.models[model_key]
                    predictions, probabilities = self._predict_with_model(model, X_test, model_key)
                    
                    if predictions is not None:
                        # Create result dataframe by merging predictions with original data
                        result_df = original_df.copy()
                        result_df[f'Bug_Predicted'] = predictions
                        
                        if probabilities is not None:
                            if len(probabilities.shape) > 1 and probabilities.shape[1] > 1:
                                # For multi-class (or binary with explicit proba of 1)
                                try:
                                    # Try to get the probability of the '1' class (index 1)
                                    proba_col = probabilities[:, 1]
                                except IndexError:
                                    # Fallback if there's only one column (e.g., raw binary output)
                                    proba_col = probabilities
                                result_df[f'Bug_Probability'] = proba_col
                            elif probabilities.ndim == 1:
                                # For binary classification where output is a single probability
                                result_df[f'Bug_Probability'] = probabilities
                        
                        # Store and display results
                        self.prediction_results[model_key] = result_df
                        table = getattr(self, f'table_{model_key}')
                        self.load_df_to_table(result_df, table)
                        
                        successful_predictions += 1
                    else:
                        logging.warning("Skipping display for %s due to prediction failure.", model_key)
            
            if successful_predictions == 0:
                QMessageBox.critical(self, "Prediction Failed", "All model predictions failed. Please check your model files, input data, and feature alignment.")
                return
            
            # Save temporary CSV files
            self.save_temp_csvs()
            
            # Show results
            self.preview_box.setVisible(True)
            QMessageBox.information(self, "Success", f"Prediction completed successfully! {successful_predictions}/{len(self.models)} models executed.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"A critical error occurred during prediction: {str(e)}")
            logging.error("Default prediction failed: %s", str(e), exc_info=True)
        finally:
            # End UI feedback
            self.progress_bar.setVisible(False)
            self.run_btn.setEnabled(True)
            self.run_btn.setText("🚀 Run Default Prediction")

    def load_df_to_table(self, df: pd.DataFrame, table: QTableWidget):
        """Load DataFrame data into the specified table widget (limited to 500 rows)."""
        if df is None:
            table.setRowCount(0)
            table.setColumnCount(0)
            return

        preview_rows = min(500, len(df))
        table.setRowCount(preview_rows)
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(df.columns.tolist())
        
        # Optimize filling the table (use iterrows only on the slice being displayed)
        for row_idx in range(preview_rows):
            row_data = df.iloc[row_idx]
            for col_idx, value in enumerate(row_data):
                # Format floating point numbers nicely
                if isinstance(value, float):
                    display_value = f"{value:.4f}"
                else:
                    display_value = str(value)
                
                table.setItem(row_idx, col_idx, QTableWidgetItem(display_value))

        # Only resize visible columns
        table.resizeColumnsToContents()

    def save_temp_csvs(self):
        """Save all prediction results to temporary CSV files for later download."""
        temp_dir = Path(tempfile.gettempdir())
        
        model_filenames = {
            'p1_local': 'p1_local_predictions.csv',
            'p2_local': 'p2_local_predictions.csv',
            'p3_local': 'p3_local_predictions.csv',
            'global': 'global_predictions.csv'
        }
        
        for model_key, filename in model_filenames.items():
            if model_key in self.prediction_results:
                temp_path = temp_dir / filename
                self.prediction_results[model_key].to_csv(temp_path, index=False)
                self.temp_csv_paths[model_key] = str(temp_path)

    def download_csv(self, model_key: str):
        """Download the prediction results as a CSV file from the temporary path."""
        if model_key not in self.prediction_results or model_key not in self.temp_csv_paths:
            QMessageBox.critical(self, "Error", "No prediction results available. Please run a prediction first.")
            return

        temp_path = Path(self.temp_csv_paths[model_key])
        if not temp_path.exists():
            QMessageBox.critical(self, "Error", "Temporary result file not found.")
            return

        model_name = self.model_info.get(model_key, {}).get('name', 'Model').replace(' ', '_')
        default_name = f"{model_name}_predictions.csv"
        
        # Open file dialog to choose save location
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV File",
            default_name,
            "CSV Files (*.csv)"
        )
        
        if save_path:
            try:
                shutil.copy(temp_path, save_path)
                QMessageBox.information(self, "Success", f"CSV file saved to:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save CSV: {e}")

if __name__ == '__main__':
    # This block allows you to run the widget in isolation for testing
    app = QApplication([])
    
    # Create dummy model files for testing if they don't exist
    # NOTE: In a real app, you would train and save these models beforehand.
    # For a runnable example, we create simple placeholder files.
    temp_dir = Path(__file__).parent
    model_keys = ['p1_local', 'p2_local', 'p3_local', 'global']
    for key in model_keys:
        temp_file = temp_dir / f"{key}_local_model.joblib"
        if not temp_file.exists():
            # Create a dummy model (e.g., a simple list) and save it
            joblib.dump(['dummy_model'], temp_file) 
            logging.info(f"Created dummy model file for {key}.")
        
    window = DefaultModelTab()
    window.setWindowTitle("Enhanced Default Model Prediction Tab")
    window.resize(1000, 800)
    window.show()
    app.exec_()