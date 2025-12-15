import os
import json
import pandas as pd
import numpy as np
import cloudpickle # Used for loading models and scalers
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QFileDialog, 
    QFormLayout, QLabel, QGroupBox, QTextEdit, QMessageBox, QDialog, 
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt5.QtGui import QPixmap

# NOTE: Assuming ui.components.csv_analytics_dialog is correctly imported
from ui.components.csv_analytics_dialog import CSVAnalyticsDialog 

# =========================================================================
# === 1. WDP-ML UTILITIES (LOADING ASSETS) ================================
# =========================================================================

def load_wpdp_model(model_dir):
    """Loads the trained model from the given directory using cloudpickle."""
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = cloudpickle.load(f)
        return model
    except Exception as e:
        raise IOError(f"Failed to load model from {model_path}. Error: {e}")

def load_wpdp_scaler(model_dir):
    """Loads the feature scaler from the given directory (returns None if not found)."""
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        # This is expected if normalization was not used during training
        return None 
    try:
        with open(scaler_path, 'rb') as f:
            scaler = cloudpickle.load(f)
        return scaler
    except Exception as e:
        raise IOError(f"Failed to load scaler from {scaler_path}. Error: {e}")

def load_wpdp_features(model_dir):
    """Loads the list of features used for training."""
    features_path = os.path.join(model_dir, "features.json")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature list (features.json) not found at {features_path}. This file is required for inference.")
    try:
        with open(features_path, 'r') as f:
            features = json.load(f)
        return features
    except Exception as e:
        raise IOError(f"Failed to load features from {features_path}. Error: {e}")

# =========================================================================
# === 2. MODEL PREDICTION WORKER (IMPLEMENTATION) =========================
# =========================================================================

class PredictionWorkerSignals(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str) # Added for feedback

class ModelPredictionWorker(QRunnable):
    """Worker to handle the prediction logic on a separate thread."""
    def __init__(self, model_dir, data_path, output_path):
        super().__init__()
        self.model_dir = model_dir
        self.data_path = data_path
        self.output_path = output_path
        self.signals = PredictionWorkerSignals()

    def run(self):
        self.signals.progress.emit("Starting WDP-ML inference...")
        try:
            # 1. Load Assets (Model, Scaler, Features)
            self.signals.progress.emit("Loading model assets...")
            model = load_wpdp_model(self.model_dir)
            scaler = load_wpdp_scaler(self.model_dir) # Can be None
            final_features = load_wpdp_features(self.model_dir)
            
            # 2. Load Unlabeled Data
            self.signals.progress.emit(f"Reading data from: {self.data_path}")
            data_to_predict = pd.read_csv(self.data_path)
            
            if data_to_predict.empty:
                raise ValueError("Input data is empty.")


            ignore_cols = ['function _name', 'filepath', 'commit_hash']
            data_to_predict = data_to_predict.drop(columns=[col for col in ignore_cols if col in data_to_predict.columns], errors='ignore')
            
            # 3. Select Features and Handle Missing Columns
            missing_features = [f for f in final_features if f not in data_to_predict.columns]
            if missing_features:
                 raise ValueError(f"Input CSV is missing {len(missing_features)} required features, e.g., {missing_features[:3]}")

            # Filter the input data to ONLY contain the features the model was trained on
            # This also ensures the correct order of features (column order matters!)
            X_pred = data_to_predict[final_features].copy()
            
            # 4. Apply Scaling (Transformation)
            X_input = X_pred.values # Start with numpy array of features
            if scaler:
                self.signals.progress.emit("Applying feature scaling (transform)...")
                X_input = scaler.transform(X_input) # Apply the fitted scaler's transformation

            # 5. Generate Predictions
            self.signals.progress.emit("Generating predictions...")
            predictions = model.predict(X_input)
            
            # 6. Integrate Results and Save
            data_to_predict['Prediction (Bug=1)'] = predictions.astype(int)
            
            self.signals.progress.emit(f"Saving results to: {self.output_path}")
            data_to_predict.to_csv(self.output_path, index=False)
            
            self.signals.finished.emit({
                'output_path': self.output_path,
                'rows_predicted': len(data_to_predict)
            })

        except FileNotFoundError as e:
            self.signals.error.emit(f"Inference Setup Error: Required file not found: {str(e)}")
        except Exception as e:
            self.signals.error.emit(f"Prediction Error: {type(e).__name__}: {str(e)}")

# =========================================================================
# === 3. CSV VIEWER DIALOG (NEW) ==========================================
# =========================================================================

class CsvViewerDialog(QDialog):
    """Simple dialog to display the head of the output CSV."""
    def __init__(self, file_path, parent=None, head_size=10):
        super().__init__(parent)
        self.setWindowTitle(f"Prediction Output Preview: {os.path.basename(file_path)}")
        self.setGeometry(100, 100, 800, 400)
        
        layout = QVBoxLayout(self)
        
        try:
            df = pd.read_csv(file_path)
            df_head = df.head(head_size)
            
            table = QTableWidget()
            table.setRowCount(df_head.shape[0])
            table.setColumnCount(df_head.shape[1])
            table.setHorizontalHeaderLabels(df_head.columns)
            
            for i in range(df_head.shape[0]):
                for j in range(df_head.shape[1]):
                    value = str(df_head.iloc[i, j])
                    item = QTableWidgetItem(value)
                    # Highlight the prediction column
                    if df_head.columns[j] == 'Prediction (Bug=1)':
                         item.setBackground(Qt.yellow)
                    table.setItem(i, j, item)
            
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            layout.addWidget(table)
            
        except Exception as e:
            layout.addWidget(QLabel(f"Error loading CSV for display: {e}"))
            
        self.setLayout(layout)

# =========================================================================
# === 4. UI INTEGRATION (PredictionScenarioWV) ============================
# =========================================================================

class PredictionScenarioWV(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threadpool = QThreadPool()
        self.cm_plot_path = None
        self.init_ui()
    # ... (init_ui remains the same) ...
# I will rewrite the class definition for completeness and place the new methods.

    def init_ui(self):
        vbox = QVBoxLayout(self)
        vbox.setSpacing(10)

        # --- A. Model Load Configuration ---
        model_load_group = QGroupBox("Load Trained Model")
        load_form = QFormLayout()
        
        self.model_dir_input = QLineEdit()
        self.model_dir_input.setPlaceholderText("Select the folder where the model and log were saved.")
        self.browse_model_btn = QPushButton("Browse Folder")
        self.browse_model_btn.clicked.connect(self.select_model_folder)
        
        h_model = QHBoxLayout()
        h_model.addWidget(self.model_dir_input)
        h_model.addWidget(self.browse_model_btn)
        load_form.addRow(QLabel("Model Folder:"), h_model)
        
        self.load_model_btn = QPushButton("Load Model & Display Training Results")
        self.load_model_btn.clicked.connect(self.load_model_summary)
        load_form.addRow(self.load_model_btn)
        
        model_load_group.setLayout(load_form)
        vbox.addWidget(model_load_group)

        # --- B. Training Summary Display ---
        self.summary_group = QGroupBox("Training Run Summary")
        self.summary_group.setVisible(False)
        summary_layout = QVBoxLayout(self.summary_group)
        
        self.metrics_label = QLabel("Metrics: N/A | Model: N/A | Features: N/A")
        self.metrics_label.setStyleSheet("font-weight: bold;")
        summary_layout.addWidget(self.metrics_label)
        
        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setFixedHeight(120) 
        self.config_text.setPlaceholderText("Hyperparameters and run configuration will appear here...")
        summary_layout.addWidget(QLabel("Configuration:"))
        summary_layout.addWidget(self.config_text)
        
        self.cm_plot_display = QLabel("Confusion Matrix Plot (from Training) will appear here.")
        self.cm_plot_display.setAlignment(Qt.AlignCenter)
        self.cm_plot_display.setStyleSheet("border: 1px solid lightgray; padding: 5px;")
        summary_layout.addWidget(self.cm_plot_display)
        
        vbox.addWidget(self.summary_group)

        # --- C. Prediction Configuration ---
        predict_group = QGroupBox("Prediction/Inference")
        predict_form = QFormLayout()
        
        self.data_input = QLineEdit()
        self.data_input.setPlaceholderText("Select the unlabeled CSV file for prediction.")
        self.browse_data_btn = QPushButton("Browse CSV")
        self.browse_data_btn.clicked.connect(self.select_unlabeled_data)
        
        h_data = QHBoxLayout()
        h_data.addWidget(self.data_input)
        h_data.addWidget(self.browse_data_btn)
        predict_form.addRow(QLabel("Unlabeled Data:"), h_data)
        
        self.output_file_input = QLineEdit()
        self.output_file_input.setPlaceholderText("Enter output CSV file path for predictions.")
        self.browse_output_btn = QPushButton("Browse Output Path")
        self.browse_output_btn.clicked.connect(self.select_output_file)
        
        h_output = QHBoxLayout()
        h_output.addWidget(self.output_file_input)
        h_output.addWidget(self.browse_output_btn)
        predict_form.addRow(QLabel("Save Predictions To:"), h_output)
        
        self.predict_btn = QPushButton("Run Prediction")
        self.predict_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.predict_btn.clicked.connect(self.start_prediction)
        
        predict_group.setLayout(predict_form)
        vbox.addWidget(predict_group)
        vbox.addWidget(self.predict_btn)

        # --- D. Prediction Results ---
        self.results_group = QGroupBox("Prediction Results")
        self.results_group.setVisible(False)
        results_layout = QVBoxLayout(self.results_group)
        
        self.output_path_label = QLabel("Output CSV: N/A")
        self.output_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        results_layout.addWidget(self.output_path_label)
        
        # New buttons for viewing output
        h_results_btns = QHBoxLayout()
        self.view_csv_btn = QPushButton("Preview Prediction CSV")
        self.view_csv_btn.clicked.connect(self.display_output_csv)
        h_results_btns.addWidget(self.view_csv_btn)

        self.analytics_btn = QPushButton("View Detailed Analytics on Predictions")
        self.analytics_btn.clicked.connect(self.view_detailed_analytics)
        h_results_btns.addWidget(self.analytics_btn)
        
        results_layout.addLayout(h_results_btns)
        
        vbox.addWidget(self.results_group)
        vbox.addStretch(1)
        
        self.set_ui_state(True)
        self.predict_btn.setEnabled(False) 

    # --- UI STATE & FILE DIALOGS (Methods remain the same) ---

    def set_ui_state(self, enabled):
        """Enable/disable main prediction UI elements."""
        self.model_dir_input.setEnabled(enabled)
        self.browse_model_btn.setEnabled(enabled)
        self.load_model_btn.setEnabled(enabled)
        self.data_input.setEnabled(enabled)
        self.browse_data_btn.setEnabled(enabled)
        self.output_file_input.setEnabled(enabled)
        self.browse_output_btn.setEnabled(enabled)
        
        # Only enable prediction button if a model summary is loaded
        if enabled and self.cm_plot_path:
            self.predict_btn.setEnabled(True)
            self.predict_btn.setText("Run Prediction")
            self.predict_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        else:
            self.predict_btn.setEnabled(False)
            self.predict_btn.setText("Waiting for Model Load...")
            self.predict_btn.setStyleSheet("background-color: gray; color: white; font-weight: bold; padding: 8px;")


    def select_model_folder(self):
        """Selects the model directory containing the log, model, and plot."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Saved Model Folder", os.getcwd())
        if dir_path:
            self.model_dir_input.setText(dir_path)
            self.summary_group.setVisible(False)
            self.set_ui_state(True) # Re-enable load button

    def select_unlabeled_data(self):
        """Selects the unlabeled CSV for inference."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Unlabeled CSV File", os.getcwd(), "CSV Files (*.csv)")
        if file_path:
            self.data_input.setText(file_path)
            # Automatically suggest an output file name
            dir_name = os.path.dirname(file_path)
            base_name = os.path.basename(file_path).replace('.csv', '_predictions.csv')
            self.output_file_input.setText(os.path.join(dir_name, base_name))

    def select_output_file(self):
        """Selects the location to save the prediction CSV."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Prediction Results As", os.getcwd(), "CSV Files (*.csv)")
        if file_path:
            # Ensure it ends with .csv
            if not file_path.lower().endswith('.csv'):
                file_path += '.csv'
            self.output_file_input.setText(file_path)

    # --- MODEL LOADING LOGIC (Remains the same) ---

    def load_model_summary(self):
        """Loads the metrics, config, and CM plot from the selected directory."""
        model_dir = self.model_dir_input.text().strip()
        
        if not os.path.isdir(model_dir):
            QMessageBox.warning(self, "Load Error", "Please select a valid model folder.")
            return

        log_path = os.path.join(model_dir, "run_log.json") 
        cm_plot_path = os.path.join(model_dir, "confusion_matrix.png") 
        
        self.cm_plot_path = None 
        log_data = None
        
        # 1. Load Log
        if os.path.isfile(log_path):
            try:
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
            except Exception as e:
                QMessageBox.critical(self, "Log Error", f"Failed to read run log file: {str(e)}")
                return
        else:
            QMessageBox.warning(self, "Load Error", f"Run log not found at: {log_path}")
            return
        
        # 2. Update Summary UI
        self._update_summary_ui(log_data, cm_plot_path)

        # 3. Enable Prediction UI
        self.cm_plot_path = cm_plot_path
        self.set_ui_state(True)

    def _update_summary_ui(self, log_data, cm_plot_path):
        """Displays the loaded training summary data."""
        results = log_data.get('results', {})
        config = log_data.get('configuration', {})
        
        def safe_format_metric(key, metrics_dict):
            raw_value = metrics_dict.get(key, 'N/A') 
            try:
                num_value = float(raw_value)
                return f"{num_value:.4f}"
            except (ValueError, TypeError):
                return str(raw_value)

        acc_str = safe_format_metric('Accuracy', results)
        f1_str = safe_format_metric('F1 Score', results)
        
        model_name = config.get('model', 'N/A')
        feat_count = results.get('Feature Count', 'N/A') 
        
        metrics_summary = (
            f"Model: {model_name} | Accuracy: {acc_str} | F1 Score: {f1_str} | Features: {feat_count}"
        )
        self.metrics_label.setText(metrics_summary)
        
        hyperparams = config.get('hyperparams', {}) 
        display_config = {
            "model": config.get('model', 'N/A'),
            "split_ratio": config.get('split_ratio', 'N/A'),
            "normalize": config.get('normalize', 'N/A'),
            "smote": config.get('smote', 'N/A'),
            "hyperparams": hyperparams
        }
        self.config_text.setText(json.dumps(display_config, indent=2))
        
        # Display CM Plot
        self.cm_plot_display.setText("Loading Confusion Matrix...")
        if os.path.isfile(cm_plot_path):
            pixmap = QPixmap(cm_plot_path)
            self.cm_plot_display.setPixmap(pixmap.scaled(
                350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        else:
            self.cm_plot_display.setText(f"CM Plot file not found at: {cm_plot_path}")
            
        self.summary_group.setVisible(True)

    # --- PREDICTION WORKFLOW (UPDATED TO CONNECT PROGRESS) ---

    def start_prediction(self):
        """Initiates the prediction process on the selected unlabeled data."""
        model_dir = self.model_dir_input.text().strip()
        data_path = self.data_input.text().strip()
        output_path = self.output_file_input.text().strip()
        
        # Validation checks
        if not all([model_dir, data_path, output_path]):
            QMessageBox.warning(self, "Input Error", "Please select a Model Folder, Unlabeled Data CSV, and Output File path.")
            return
        if not os.path.isdir(model_dir) or not os.path.isfile(data_path):
            QMessageBox.warning(self, "Input Error", "Invalid Model Folder or Data CSV.")
            return
        
        self.set_ui_state(False)
        self.predict_btn.setText("Predicting... (Initializing)")
        self.results_group.setVisible(False)
        
        worker = ModelPredictionWorker(model_dir, data_path, output_path)
        worker.signals.finished.connect(self.prediction_finished)
        worker.signals.error.connect(self.prediction_error)
        worker.signals.progress.connect(self.update_prediction_progress) # <--- NEW CONNECTION
        
        self.threadpool.start(worker)
        
    def update_prediction_progress(self, message):
        """Updates the prediction button text with current progress."""
        self.predict_btn.setText(f"Predicting... ({message})")

    def prediction_finished(self, results):
        """Handles the results after prediction worker completes."""
        output_path = results['output_path']
        rows_predicted = results['rows_predicted']
        
        self.output_path_label.setText(f"Output CSV: {output_path}")
        self.results_group.setVisible(True)
        
        QMessageBox.information(self, "Prediction Complete", 
                                f"Prediction saved successfully.\nRows Predicted: {rows_predicted:,}")
        
        self.set_ui_state(True)
        self.predict_btn.setText("Run Prediction (Model Loaded)")


    def prediction_error(self, message):
        """Handles errors from the prediction worker."""
        QMessageBox.critical(self, "Prediction Error", message)
        self.set_ui_state(True)
        self.predict_btn.setText("Run Prediction (ERROR)")
        self.predict_btn.setStyleSheet("background-color: red; color: white; font-weight: bold; padding: 8px;")

    # --- ANALYTICS/VIEW ---

    def display_output_csv(self):
        """Opens a new dialog to display the head of the output CSV."""
        file_path = self.output_file_input.text().strip()
        
        if not os.path.isfile(file_path):
            QMessageBox.warning(self, "File Error", "Prediction output file not found. Run prediction first.")
            return

        try:
            dialog = CsvViewerDialog(file_path, self)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "View Error", f"Failed to preview CSV: {str(e)}")


    def view_detailed_analytics(self):
        """Opens the CSVAnalyticsDialog for the prediction output."""
        file_path = self.output_file_input.text().strip()
        
        if not os.path.isfile(file_path):
            QMessageBox.warning(self, "File Error", "Prediction output file not found. Run prediction first.")
            return

        try:
            # Reuses the CSVAnalyticsDialog component
            dialog = CSVAnalyticsDialog(file_path, self)
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Analytics Error", f"Failed to load detailed analytics: {str(e)}")