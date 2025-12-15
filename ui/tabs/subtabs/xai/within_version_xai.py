import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QFileDialog, 
    QFormLayout, QLabel, QGroupBox, QTextEdit, QMessageBox, QDialog, 
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QDoubleSpinBox,
    QSpinBox, QScrollArea, QFrame, QSizePolicy, QProgressBar, QCheckBox,
    QTextBrowser
)
from PyQt5.QtCore import Qt, QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt5.QtGui import QPixmap

from core.xai_wvwp import wvwp_shap_single_csv 

# Import components
from ui.components.csv_analytics_dialog import CSVAnalyticsDialog
# from ui.components.csv_viewer_dialog import CsvViewerDialog


class WithinVersionXAISignals(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

class WithinVersionXAIWorker(QRunnable):
    """Worker to handle within-version XAI analysis."""
    def __init__(self, csv_path, model_class, model_params, 
                 train_test_split_ratio, shap_threshold, smote_random_state):
        super().__init__()
        self.csv_path = csv_path
        self.model_class = model_class
        self.model_params = model_params
        self.train_test_split_ratio = train_test_split_ratio
        self.shap_threshold = shap_threshold
        self.smote_random_state = smote_random_state
        self.signals = WithinVersionXAISignals()

    def run(self):
        try:
            self.signals.progress.emit("Starting XAI analysis...")
            
            # Prepare parameters for the core function
            params = {
                "csv_path": self.csv_path,
                "model_class": self.model_class,
                "model_params": self.model_params,
                "train_test_split_ratio": self.train_test_split_ratio,
                "shap_threshold": self.shap_threshold,
                "smote_random_state": self.smote_random_state,
                # Default parameters that match your function signature
                "feature_cols": None,
                "normalize": False,
                "apply_smote": True if self.smote_random_state is not None else False,
                "sampling_strategy": 0.5,
                "random_state": 42
            }
            
            self.signals.progress.emit("Loading and preprocessing data...")
            
            # Call the actual core function
            metrics_df, features_df, shap_df = wvwp_shap_single_csv(**params)
            
            self.signals.progress.emit("Analysis complete! Preparing results...")
            
            # Return results as a dictionary
            result = {
                'metrics_df': metrics_df,
                'features_df': features_df,
                'shap_df': shap_df,
                'analysis_type': 'within_version'
            }
            
            self.signals.finished.emit(result)
            
        except FileNotFoundError as e:
            self.signals.error.emit(f"File not found: {str(e)}")
        except ValueError as e:
            self.signals.error.emit(f"Data validation error: {str(e)}")
        except ImportError as e:
            self.signals.error.emit(f"Missing dependency: {str(e)}")
        except Exception as e:
            self.signals.error.emit(f"XAI Analysis Error: {type(e).__name__}: {str(e)}")

# =========================================================================
# === UI FOR WITHIN VERSION XAI ===========================================
# =========================================================================

class WithinVersionXAI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.threadpool = QThreadPool()
        self.current_csv_path = None
        self.results = {
            'metrics_df': None,
            'features_df': None,
            'shap_df': None
        }
        self.init_ui()

    def init_ui(self):
        vbox = QVBoxLayout(self)
        vbox.setSpacing(15)  # Increased spacing
        vbox.setContentsMargins(20, 20, 20, 20)  # Increased margins

        # --- A. CSV File Configuration ---
        csv_group = QGroupBox("CSV File Configuration")
        csv_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        csv_form = QFormLayout()
        csv_form.setSpacing(10)

        # CSV file input
        self.csv_input = QLineEdit()
        self.csv_input.setPlaceholderText("Select CSV file for analysis")
        self.browse_csv_btn = QPushButton("Browse CSV")
        self.browse_csv_btn.clicked.connect(self.select_csv_file)
        
        h_csv = QHBoxLayout()
        h_csv.addWidget(self.csv_input)
        h_csv.addWidget(self.browse_csv_btn)
        csv_form.addRow(QLabel("Data CSV:"), h_csv)

        # CSV analytics button
        self.csv_analytics_btn = QPushButton("Show CSV Analytics")
        self.csv_analytics_btn.clicked.connect(self.show_csv_analytics)
        self.csv_analytics_btn.setStyleSheet("""
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
        csv_form.addRow(self.csv_analytics_btn)

        # Train/test split
        self.train_split_spin = QDoubleSpinBox()
        self.train_split_spin.setRange(0.1, 0.9)
        self.train_split_spin.setValue(0.7)
        self.train_split_spin.setSingleStep(0.1)
        self.train_split_spin.setDecimals(2)
        csv_form.addRow("Train/Test Split Ratio:", self.train_split_spin)

        csv_group.setLayout(csv_form)
        vbox.addWidget(csv_group)

        # --- B. Model Configuration ---
        model_group = QGroupBox("Machine Learning Model Configuration")
        model_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        model_form = QFormLayout()
        model_form.setSpacing(10)

        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Random Forest", "XGBoost", "Decision Tree", 
            "Logistic Regression", "Support Vector Machine", "Neural Network"
        ])
        self.model_combo.currentTextChanged.connect(self.update_model_parameters)
        model_form.addRow("Select Model:", self.model_combo)

        # Model parameters container
        self.params_container = QWidget()
        self.params_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.params_layout = QFormLayout(self.params_container)
        model_form.addRow(self.params_container)

        # Initialize with default parameters
        self.param_widgets = {}
        self.update_model_parameters(self.model_combo.currentText())

        model_group.setLayout(model_form)
        vbox.addWidget(model_group)

        # --- C. XAI Configuration ---
        xai_group = QGroupBox("XAI Configuration")
        xai_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        xai_form = QFormLayout()
        xai_form.setSpacing(10)

        # SHAP threshold
        self.shap_threshold_spin = QDoubleSpinBox()
        self.shap_threshold_spin.setRange(0.1, 1.0)
        self.shap_threshold_spin.setValue(0.75)
        self.shap_threshold_spin.setSingleStep(0.05)
        self.shap_threshold_spin.setDecimals(2)
        xai_form.addRow("SHAP Importance Threshold:", self.shap_threshold_spin)

        # SMOTE checkbox
        self.smote_checkbox = QCheckBox("Apply SMOTE for class balancing")
        self.smote_checkbox.setChecked(True)
        xai_form.addRow(self.smote_checkbox)

        # SMOTE random state
        self.smote_random_spin = QSpinBox()
        self.smote_random_spin.setRange(0, 100)
        self.smote_random_spin.setValue(42)
        xai_form.addRow("SMOTE Random State:", self.smote_random_spin)

        xai_group.setLayout(xai_form)
        vbox.addWidget(xai_group)

        # --- D. Run Analysis Button ---
        self.run_btn = QPushButton("Run XAI Analysis")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px;
                font-weight: bold;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.run_btn.clicked.connect(self.start_analysis)
        vbox.addWidget(self.run_btn)

        # --- E. Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        vbox.addWidget(self.progress_bar)

        # --- F. Results Display ---
        self.results_group = QGroupBox("XAI Analysis Results")
        self.results_group.setVisible(False)
        self.results_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_layout = QVBoxLayout(self.results_group)

        # Model Metrics Results
        metrics_frame = QFrame()
        metrics_frame.setFrameStyle(QFrame.StyledPanel)
        metrics_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        metrics_layout = QVBoxLayout(metrics_frame)

        metrics_header = QHBoxLayout()
        metrics_header.addWidget(QLabel("Model Performance Metrics"))
        
        # FIX: Ensure this line correctly assigns the button as an attribute
        self.metrics_download_btn = QPushButton("Download Metrics CSV") 
        self.metrics_download_btn.clicked.connect(self.download_metrics)
        self.metrics_download_btn.setEnabled(False)
        metrics_header.addWidget(self.metrics_download_btn)
        metrics_header.addStretch()
        
        metrics_layout.addLayout(metrics_header)

        self.metrics_table = QTableWidget()
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.setMaximumHeight(150)
        metrics_layout.addWidget(self.metrics_table)

        results_layout.addWidget(metrics_frame)

        # Selected Features Results
        features_frame = QFrame()
        features_frame.setFrameStyle(QFrame.StyledPanel)
        features_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        features_layout = QVBoxLayout(features_frame)

        features_header = QHBoxLayout()
        features_header.addWidget(QLabel("Top Important Features (SHAP-based)"))
        
        # FIX: Ensure this line correctly assigns the button as an attribute
        self.features_download_btn = QPushButton("Download Features CSV") 
        self.features_download_btn.clicked.connect(self.download_features)
        self.features_download_btn.setEnabled(False)
        features_header.addWidget(self.features_download_btn)
        features_header.addStretch()
        
        features_layout.addLayout(features_header)

        self.features_table = QTableWidget()
        self.features_table.setAlternatingRowColors(True)
        self.features_table.horizontalHeader().setStretchLastSection(True)
        self.features_table.setEditTriggers(QTableWidget.NoEditTriggers)
        features_layout.addWidget(self.features_table)

        results_layout.addWidget(features_frame)

        # Additional actions
        actions_layout = QHBoxLayout()
        
        # FIX: Ensure all additional buttons are also assigned as attributes
        self.view_shap_btn = QPushButton("View SHAP Summary Plot")
        self.view_shap_btn.clicked.connect(self.view_shap_summary)
        self.view_shap_btn.setEnabled(False)
        actions_layout.addWidget(self.view_shap_btn)
        
        self.view_analysis_btn = QPushButton("View Detailed Analysis")
        self.view_analysis_btn.clicked.connect(self.view_detailed_analysis)
        self.view_analysis_btn.setEnabled(False)
        actions_layout.addWidget(self.view_analysis_btn)
        
        self.download_shap_btn = QPushButton("Download Feature Selection CSV")
        self.download_shap_btn.clicked.connect(self.download_shap_values)
        self.download_shap_btn.setEnabled(False)
        actions_layout.addWidget(self.download_shap_btn)
        
        results_layout.addLayout(actions_layout)

        vbox.addWidget(self.results_group)
        vbox.addStretch() # Use stretch to push content upwards

        # Set initial UI state
        self.set_ui_state(True)
        self.run_btn.setEnabled(False)  # Disabled until CSV is selected

    def update_model_parameters(self, model_name):
        """Update parameter controls based on selected model."""
        # Clear existing widgets
        for i in reversed(range(self.params_layout.count())):
            widget = self.params_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        self.param_widgets.clear()
        
        # Add parameters based on model
        if model_name == "Random Forest":
            self.add_int_param("n_estimators", "Number of Trees", 10, 500, 100)
            self.add_int_param("max_depth", "Max Depth", 1, 50, 10)
            self.add_int_param("min_samples_split", "Min Samples Split", 2, 20, 2)
            self.add_int_param("min_samples_leaf", "Min Samples Leaf", 1, 20, 1)
            
        elif model_name == "XGBoost":
            self.add_double_param("learning_rate", "Learning Rate", 0.01, 1.0, 0.3)
            self.add_int_param("max_depth", "Max Depth", 1, 20, 6)
            self.add_int_param("n_estimators", "Number of Trees", 10, 1000, 100)
            self.add_double_param("subsample", "Subsample", 0.1, 1.0, 1.0)
            
        elif model_name == "Decision Tree":
            self.add_int_param("max_depth", "Max Depth", 1, 50, 10)
            self.add_int_param("min_samples_split", "Min Samples Split", 2, 20, 2)
            self.add_int_param("min_samples_leaf", "Min Samples Leaf", 1, 20, 1)
            self.add_combo_param("criterion", "Criterion", ["gini", "entropy"])
            
        elif model_name == "Logistic Regression":
            self.add_double_param("C", "Regularization (C)", 0.1, 10.0, 1.0)
            self.add_combo_param("penalty", "Penalty", ["l1", "l2", "elasticnet"])
            self.add_int_param("max_iter", "Max Iterations", 100, 5000, 1000)
            
        elif model_name == "Support Vector Machine":
            self.add_double_param("C", "Regularization (C)", 0.1, 10.0, 1.0)
            self.add_combo_param("kernel", "Kernel", ["linear", "rbf", "poly"])
            self.add_double_param("gamma", "Gamma", 0.001, 10.0, 0.1)
            
        elif model_name == "Neural Network":
            self.add_combo_param("hidden_layer_sizes", "Hidden Layers", 
                                ["(50,)", "(100,)", "(50,50)", "(100,50)"])
            self.add_double_param("alpha", "Alpha", 0.0001, 1.0, 0.0001)
            self.add_combo_param("activation", "Activation", ["relu", "tanh", "logistic"])

    def add_int_param(self, name, label, min_val, max_val, default_val):
        """Add integer parameter control."""
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default_val)
        self.params_layout.addRow(f"{label}:", spinbox)
        self.param_widgets[name] = spinbox

    def add_double_param(self, name, label, min_val, max_val, default_val):
        """Add double parameter control."""
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default_val)
        spinbox.setSingleStep(0.01)
        self.params_layout.addRow(f"{label}:", spinbox)
        self.param_widgets[name] = spinbox

    def add_combo_param(self, name, label, options):
        """Add combobox parameter control."""
        combobox = QComboBox()
        combobox.addItems(options)
        self.params_layout.addRow(f"{label}:", combobox)
        self.param_widgets[name] = combobox

    def set_ui_state(self, enabled):
        """Enable/disable UI elements during analysis."""
        self.csv_input.setEnabled(enabled)
        self.browse_csv_btn.setEnabled(enabled)
        self.csv_analytics_btn.setEnabled(enabled)
        self.train_split_spin.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.shap_threshold_spin.setEnabled(enabled)
        self.smote_checkbox.setEnabled(enabled)
        self.smote_random_spin.setEnabled(enabled)
        
        # Enable/disable parameter widgets
        for widget in self.param_widgets.values():
            widget.setEnabled(enabled)
        
        if enabled:
            self.run_btn.setText("Run XAI Analysis")
            self.run_btn.setEnabled(self.current_csv_path is not None)
        else:
            self.run_btn.setText("Analyzing...")
            self.run_btn.setEnabled(False)

    # ===== File Dialogs =====
    def select_csv_file(self):
        """Select CSV file for analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", os.getcwd(), "CSV Files (*.csv)"
        )
        if file_path:
            self.current_csv_path = file_path
            self.csv_input.setText(file_path)
            self.run_btn.setEnabled(True)

    def show_csv_analytics(self):
        """Show analytics for selected CSV file."""
        if not self.current_csv_path or not os.path.exists(self.current_csv_path):
            QMessageBox.warning(self, "File Error", "Please select a CSV file first.")
            return
        
        try:
            dialog = CSVAnalyticsDialog(self.current_csv_path, self)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Analytics Error", f"Failed to load analytics: {str(e)}")

    # ===== Analysis Logic =====
    def get_model_class_and_params(self):
        """Get the actual model class and parameters from UI."""
        model_name = self.model_combo.currentText()
        
        # Import model classes
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        
        model_map = {
            "Random Forest": RandomForestClassifier,
            "XGBoost": XGBClassifier,
            "Decision Tree": DecisionTreeClassifier,
            "Logistic Regression": LogisticRegression,
            "Support Vector Machine": SVC,
            "Neural Network": MLPClassifier
        }
        
        model_class = model_map.get(model_name, RandomForestClassifier)
        
        # Extract parameters
        model_params = {}
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, QSpinBox):
                model_params[param_name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                model_params[param_name] = widget.value()
            elif isinstance(widget, QComboBox):
                value = widget.currentText()
                # Convert string to tuple for hidden layers
                if param_name == "hidden_layer_sizes":
                    try:
                        value = eval(value)
                    except:
                        pass
                model_params[param_name] = value
        
        return model_class, model_params

    def validate_inputs(self):
        """Validate all inputs before analysis."""
        if not self.current_csv_path:
            QMessageBox.warning(self, "Input Error", "Please select a CSV file.")
            return False
        
        if not os.path.exists(self.current_csv_path):
            QMessageBox.warning(self, "Input Error", f"CSV file not found: {self.current_csv_path}")
            return False
        
        return True

    def start_analysis(self):
        """Start the XAI analysis process."""
        if not self.validate_inputs():
            return
        
        # Get model class and parameters
        model_class, model_params = self.get_model_class_and_params()
        
        # Get XAI parameters
        train_test_split_ratio = self.train_split_spin.value()
        shap_threshold = self.shap_threshold_spin.value()
        smote_random_state = self.smote_random_spin.value() if self.smote_checkbox.isChecked() else None
        
        # Clear previous results
        self.clear_results()
        
        # Set UI to analyzing state
        self.set_ui_state(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Create and start worker
        worker = WithinVersionXAIWorker(
            csv_path=self.current_csv_path,
            model_class=model_class,
            model_params=model_params,
            train_test_split_ratio=train_test_split_ratio,
            shap_threshold=shap_threshold,
            smote_random_state=smote_random_state
        )
        worker.signals.finished.connect(self.analysis_finished)
        worker.signals.error.connect(self.analysis_error)
        worker.signals.progress.connect(self.update_progress)
        
        self.threadpool.start(worker)

    def update_progress(self, message):
        """Update progress bar text."""
        self.progress_bar.setFormat(f"%p - {message}")

    def analysis_finished(self, result):
        """Handle successful analysis completion."""
        self.progress_bar.setVisible(False)
        self.set_ui_state(True)
        
        # Store results
        self.results = result
        
        # Display results
        self.display_results(result)
        
        QMessageBox.information(self, "Analysis Complete", 
                               "XAI analysis completed successfully!")

    def analysis_error(self, error_message):
        """Handle analysis errors."""
        self.progress_bar.setVisible(False)
        self.set_ui_state(True)
        QMessageBox.critical(self, "Analysis Error", error_message)

    def clear_results(self):
        """Clear previous results."""
        self.results_group.setVisible(False)
        self.metrics_table.clear()
        self.metrics_table.setRowCount(0)
        self.metrics_table.setColumnCount(0)
        self.features_table.clear()
        self.features_table.setRowCount(0)
        self.features_table.setColumnCount(0)
        self.metrics_download_btn.setEnabled(False)
        self.features_download_btn.setEnabled(False)
        self.view_shap_btn.setEnabled(False)
        self.view_analysis_btn.setEnabled(False)
        self.download_shap_btn.setEnabled(False)

    def display_results(self, results):
        """Display analysis results in tables."""
        # Show results group
        self.results_group.setVisible(True)
        
        # Display metrics
        metrics_df = results.get('metrics_df')
        if metrics_df is not None and not metrics_df.empty:
            self.metrics_table.setRowCount(1)
            self.metrics_table.setColumnCount(len(metrics_df.columns))
            self.metrics_table.setHorizontalHeaderLabels(metrics_df.columns.tolist())
            
            for col_idx, column in enumerate(metrics_df.columns):
                value = metrics_df[column].iloc[0]
                if isinstance(value, (float, np.floating)):
                    display_value = f"{value:.4f}"
                else:
                    display_value = str(value)
                
                item = QTableWidgetItem(display_value)
                self.metrics_table.setItem(0, col_idx, item)
            
            self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.metrics_download_btn.setEnabled(True)
        
        # Display features
        features_df = results.get('features_df')
        if features_df is not None and not features_df.empty:
            self.features_table.setRowCount(len(features_df))
            self.features_table.setColumnCount(2)
            self.features_table.setHorizontalHeaderLabels(['Feature', 'Rank'])
            
            for row_idx, (_, row) in enumerate(features_df.iterrows()):
                feature_item = QTableWidgetItem(str(row.get('feature', '')))
                rank_item = QTableWidgetItem(str(row.get('rank', '')))
                
                self.features_table.setItem(row_idx, 0, feature_item)
                self.features_table.setItem(row_idx, 1, rank_item)
            
            self.features_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            self.features_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            self.features_download_btn.setEnabled(True)
        
        # Enable additional buttons
        self.view_shap_btn.setEnabled(True)
        self.view_analysis_btn.setEnabled(True)
        self.download_shap_btn.setEnabled(True)

    # ===== Download Methods =====
    def download_metrics(self):
        """Download metrics results as CSV."""
        if self.results.get('metrics_df') is not None:
            self.save_dataframe(self.results['metrics_df'], "within_version_metrics.csv")

    def download_features(self):
        """
        Download the ranked features (derived from SHAP values) as a timestamped CSV.
        The resulting CSV's first column contains the feature names.
        """
        feature_df = self.results.get('features_df')
        
        if feature_df is not None and not feature_df.empty:
            
            # 1. Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"feature_selection_xai_{timestamp}.csv"
            
            self.save_dataframe(feature_df, default_name)
        else:
            QMessageBox.warning(self, "Download Error", "Feature ranking data not available.")

    # NOTE: You should consider removing the old download_shap_values method 
    # and renaming the button's clicked signal to use this new method.

    def download_shap_values(self):
        """Download full SHAP values as CSV."""
        feature_df = self.results.get('features_df').iloc[:,0]
        
        if feature_df is not None and not feature_df.empty:
            
            # 1. Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"feature_selection_xai_{timestamp}.csv"
            
            self.save_dataframe(feature_df, default_name)
        else:
            QMessageBox.warning(self, "Download Error", "Feature ranking data not available.")

    def save_dataframe(self, df, default_name):
        """Save dataframe to CSV file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV File", default_name, "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                if not file_path.lower().endswith('.csv'):
                    file_path += '.csv'
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Save Complete", 
                                       f"File saved successfully:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save file: {str(e)}")

    # ===== View Methods =====
    def view_shap_summary(self):
        """View SHAP summary plot."""
        # This would open a dialog with SHAP summary plot
        # For now, show a message with SHAP info
        if self.results.get('shap_df') is not None:
            shap_df = self.results['shap_df']
            top_features = shap_df.head(10)
            
            message = "Top 10 Most Important Features (by SHAP):\n\n"
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                feature = row.get('feature', 'Unknown')
                importance = row.get('mean_abs_shap', 0)
                message += f"{i}. {feature}: {importance:.6f}\n"
            
            QMessageBox.information(self, "SHAP Summary", message)
        else:
            QMessageBox.information(self, "SHAP Summary", 
                                   "SHAP summary data not available.")

    def view_detailed_analysis(self):
        """View detailed analysis of the original CSV."""
        if self.current_csv_path and os.path.exists(self.current_csv_path):
            try:
                dialog = CSVAnalyticsDialog(self.current_csv_path, self)
                dialog.exec_()
            except Exception as e:
                QMessageBox.critical(self, "View Error", f"Failed to load CSV: {str(e)}")