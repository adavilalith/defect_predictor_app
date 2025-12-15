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
    QTextBrowser, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt5.QtGui import QPixmap

# Import core logic directly
from core.xai_cvwp import cvwp_shap_multiple_csv

# Import components
from ui.components.csv_analytics_dialog import CSVAnalyticsDialog

# =========================================================================
# === XAI WORKER FOR CROSS VERSION ========================================
# =========================================================================

class CrossVersionXAISignals(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

class CrossVersionXAIWorker(QRunnable):
    """Worker to handle cross-version XAI analysis."""
    def __init__(self, train_csvs, test_csvs, model_class, model_params, 
                 shap_threshold, smote_random_state):
        super().__init__()
        self.train_csvs = train_csvs
        self.test_csvs = test_csvs
        self.model_class = model_class
        self.model_params = model_params
        self.shap_threshold = shap_threshold
        self.smote_random_state = smote_random_state
        self.signals = CrossVersionXAISignals()

    def run(self):
        try:
            self.signals.progress.emit("Starting Cross-Version XAI analysis...")
            
            # Prepare parameters for the core function
            params = {
                "train_csvs": self.train_csvs,
                "test_csvs": self.test_csvs,
                "model_class": self.model_class,
                "model_params": self.model_params,
                "shap_threshold": self.shap_threshold,
                "smote_random_state": self.smote_random_state
            }
            
            self.signals.progress.emit("Loading training data...")
            
            # Call the actual core function
            metrics_df, features_df, shap_df = cvwp_shap_multiple_csv(**params)
            
            self.signals.progress.emit("Analysis complete! Preparing results...")
            
            # Return results as a dictionary
            result = {
                'metrics_df': metrics_df,
                'features_df': features_df,
                'shap_df': shap_df,
                'analysis_type': 'cross_version'
            }
            
            self.signals.finished.emit(result)
            
        except FileNotFoundError as e:
            self.signals.error.emit(f"File not found: {str(e)}")
        except ValueError as e:
            self.signals.error.emit(f"Data validation error: {str(e)}")
        except ImportError as e:
            self.signals.error.emit(f"Missing dependency: {str(e)}")
        except Exception as e:
            self.signals.error.emit(f"Cross-Version XAI Analysis Error: {type(e).__name__}: {str(e)}")

# =========================================================================
# === UI FOR CROSS VERSION XAI ============================================
# =========================================================================

class CrossVersionXAI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.threadpool = QThreadPool()
        self.current_train_files = []
        self.current_test_files = []
        self.results = {
            'metrics_df': None,
            'features_df': None,
            'shap_df': None
        }
        self.init_ui()

    def init_ui(self):
        vbox = QVBoxLayout(self)
        vbox.setSpacing(15)
        vbox.setContentsMargins(20, 20, 20, 20)

        # --- A. Training Data Configuration ---
        train_group = QGroupBox("Training Data Configuration")
        train_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        train_form = QFormLayout()
        train_form.setSpacing(10)

        # Training files input
        self.train_files_input = QLineEdit()
        self.train_files_input.setPlaceholderText("Select training CSV files")
        self.browse_train_btn = QPushButton("Browse Files")
        self.browse_train_btn.clicked.connect(lambda: self.select_files("train"))
        
        h_train = QHBoxLayout()
        h_train.addWidget(self.train_files_input)
        h_train.addWidget(self.browse_train_btn)
        train_form.addRow(QLabel("Training CSVs:"), h_train)

        # Training files list with scroll area
        train_list_scroll = QScrollArea()
        train_list_scroll.setWidgetResizable(True)
        train_list_scroll.setMaximumHeight(150)
        
        self.train_list_widget = QListWidget()
        train_list_scroll.setWidget(self.train_list_widget)
        train_form.addRow(QLabel("Selected Files:"), train_list_scroll)

        # Training analytics button
        self.train_analytics_btn = QPushButton("Show Training Data Analytics")
        self.train_analytics_btn.clicked.connect(lambda: self.show_files_analytics("train"))
        self.train_analytics_btn.setStyleSheet("""
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
        train_form.addRow(self.train_analytics_btn)

        train_group.setLayout(train_form)
        vbox.addWidget(train_group)

        # --- B. Testing Data Configuration ---
        test_group = QGroupBox("Testing Data Configuration")
        test_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        test_form = QFormLayout()
        test_form.setSpacing(10)

        # Testing files input
        self.test_files_input = QLineEdit()
        self.test_files_input.setPlaceholderText("Select testing CSV files")
        self.browse_test_btn = QPushButton("Browse Files")
        self.browse_test_btn.clicked.connect(lambda: self.select_files("test"))
        
        h_test = QHBoxLayout()
        h_test.addWidget(self.test_files_input)
        h_test.addWidget(self.browse_test_btn)
        test_form.addRow(QLabel("Testing CSVs:"), h_test)

        # Testing files list with scroll area
        test_list_scroll = QScrollArea()
        test_list_scroll.setWidgetResizable(True)
        test_list_scroll.setMaximumHeight(150)
        
        self.test_list_widget = QListWidget()
        test_list_scroll.setWidget(self.test_list_widget)
        test_form.addRow(QLabel("Selected Files:"), test_list_scroll)

        # Testing analytics button
        self.test_analytics_btn = QPushButton("Show Testing Data Analytics")
        self.test_analytics_btn.clicked.connect(lambda: self.show_files_analytics("test"))
        self.test_analytics_btn.setStyleSheet("""
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
        test_form.addRow(self.test_analytics_btn)

        test_group.setLayout(test_form)
        vbox.addWidget(test_group)

        # --- C. Model Configuration ---
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

        # --- D. XAI Configuration ---
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

        # --- E. Run Analysis Button ---
        self.run_btn = QPushButton("Run Cross-Version XAI Analysis")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 12px;
                font-weight: bold;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.run_btn.clicked.connect(self.start_analysis)
        self.run_btn.setEnabled(False)
        vbox.addWidget(self.run_btn)

        # --- F. Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        vbox.addWidget(self.progress_bar)

        # --- G. Results Display ---
        self.results_group = QGroupBox("Cross-Version XAI Analysis Results")
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
        
        self.view_shap_btn = QPushButton("View SHAP Summary")
        self.view_shap_btn.clicked.connect(self.view_shap_summary)
        self.view_shap_btn.setEnabled(False)
        actions_layout.addWidget(self.view_shap_btn)
        
        self.download_shap_btn = QPushButton("Download Feature Selection CSV")
        self.download_shap_btn.clicked.connect(self.download_shap_values)
        self.download_shap_btn.setEnabled(False)
        actions_layout.addWidget(self.download_shap_btn)
        
        results_layout.addLayout(actions_layout)

        vbox.addWidget(self.results_group)
        vbox.addStretch()

        # Set initial UI state
        self.update_validation()

    # ===== File Management Methods =====
    def select_files(self, file_type):
        """Select multiple CSV files for training or testing."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, 
            f"Select {file_type.capitalize()} CSV Files", 
            os.getcwd(), 
            "CSV Files (*.csv)"
        )
        
        if file_paths:
            if file_type == "train":
                self.current_train_files = file_paths
                self.update_file_list(self.train_list_widget, file_paths)
                self.train_files_input.setText(f"{len(file_paths)} files selected")
            else:  # test
                self.current_test_files = file_paths
                self.update_file_list(self.test_list_widget, file_paths)
                self.test_files_input.setText(f"{len(file_paths)} files selected")
            
            self.update_validation()

    def update_file_list(self, list_widget, file_paths):
        """Update the list widget with file paths."""
        list_widget.clear()
        for file_path in file_paths:
            item = QListWidgetItem(os.path.basename(file_path))
            item.setToolTip(file_path)  # Show full path on hover
            list_widget.addItem(item)

    def show_files_analytics(self, file_type):
        """Show analytics for selected files."""
        file_paths = self.current_train_files if file_type == "train" else self.current_test_files
        
        if not file_paths:
            QMessageBox.warning(self, "File Error", f"No {file_type} files selected.")
            return
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    dialog = CSVAnalyticsDialog(file_path, self)
                    dialog.exec_()
                except Exception as e:
                    QMessageBox.critical(self, "Analytics Error", f"Failed to load analytics for {file_path}: {str(e)}")
            else:
                QMessageBox.warning(self, "File Error", f"File not found: {file_path}")

    def update_validation(self):
        """Update validation message and button state."""
        train_count = len(self.current_train_files)
        test_count = len(self.current_test_files)
        
        if train_count == 0 and test_count == 0:
            self.run_btn.setEnabled(False)
        elif train_count == 0:
            self.run_btn.setEnabled(False)
        elif test_count == 0:
            self.run_btn.setEnabled(False)
        else:
            self.run_btn.setEnabled(True)

    # ===== Model Configuration Methods =====
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

    # ===== Analysis Methods =====
    def validate_inputs(self):
        """Validate all inputs before analysis."""
        if not self.current_train_files:
            QMessageBox.warning(self, "Input Error", "Please select at least one training CSV file.")
            return False
        
        if not self.current_test_files:
            QMessageBox.warning(self, "Input Error", "Please select at least one testing CSV file.")
            return False
        
        # Check if files exist
        for file_path in self.current_train_files:
            if not os.path.exists(file_path):
                QMessageBox.warning(self, "Input Error", f"Training file not found: {file_path}")
                return False
        
        for file_path in self.current_test_files:
            if not os.path.exists(file_path):
                QMessageBox.warning(self, "Input Error", f"Testing file not found: {file_path}")
                return False
        
        return True

    def set_ui_state(self, enabled):
        """Enable/disable UI elements during analysis."""
        self.train_files_input.setEnabled(enabled)
        self.browse_train_btn.setEnabled(enabled)
        self.train_analytics_btn.setEnabled(enabled)
        self.train_list_widget.setEnabled(enabled)
        
        self.test_files_input.setEnabled(enabled)
        self.browse_test_btn.setEnabled(enabled)
        self.test_analytics_btn.setEnabled(enabled)
        self.test_list_widget.setEnabled(enabled)
        
        self.model_combo.setEnabled(enabled)
        self.shap_threshold_spin.setEnabled(enabled)
        self.smote_checkbox.setEnabled(enabled)
        self.smote_random_spin.setEnabled(enabled)
        
        # Enable/disable parameter widgets
        for widget in self.param_widgets.values():
            widget.setEnabled(enabled)
        
        if enabled:
            self.run_btn.setText("Run Cross-Version XAI Analysis")
            self.run_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.update_validation()
        else:
            self.run_btn.setText("Analyzing...")
            self.run_btn.setEnabled(False)

    def start_analysis(self):
        """Start the cross-version XAI analysis process."""
        if not self.validate_inputs():
            return
        
        # Get model class and parameters
        model_class, model_params = self.get_model_class_and_params()
        
        # Get XAI parameters
        shap_threshold = self.shap_threshold_spin.value()
        smote_random_state = self.smote_random_spin.value() if self.smote_checkbox.isChecked() else None
        
        # Clear previous results
        self.clear_results()
        
        # Set UI to analyzing state
        self.set_ui_state(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Create and start worker
        worker = CrossVersionXAIWorker(
            train_csvs=self.current_train_files,
            test_csvs=self.current_test_files,
            model_class=model_class,
            model_params=model_params,
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
                               "Cross-Version XAI analysis completed successfully!")

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
        self.download_shap_btn.setEnabled(True)

    # ===== Download Methods =====
    def download_metrics(self):
        """Download metrics results as CSV."""
        if self.results.get('metrics_df') is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"cross_version_metrics_{timestamp}.csv"
            self.save_dataframe(self.results['metrics_df'], default_name)

    def download_features(self):
        """Download features results as CSV."""
        if self.results.get('features_df') is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"cross_version_features_{timestamp}.csv"
            self.save_dataframe(self.results['features_df'], default_name)

    def download_shap_values(self):
        """Download full SHAP values as CSV."""
        feature_df = self.results.get('features_df')
        
        if feature_df is not None and not feature_df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"feature_selection_cross_version_xai_{timestamp}.csv"
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