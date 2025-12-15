import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QFileDialog, 
    QProgressBar, QMessageBox, QScrollArea, QTableWidget, QTableWidgetItem,
    QSizePolicy, QFormLayout, QLabel, QComboBox, QSpinBox, QCheckBox, QGroupBox,
    QGridLayout, QSlider, QTextEdit, QDoubleSpinBox
)
from PyQt5.QtCore import pyqtSignal, QObject, QRunnable, QThreadPool, Qt
from PyQt5.QtGui import QPixmap 
import json 
import pandas as pd 
import csv 

from core.model_configs import MODEL_CONFIGS 
from core.wpdp_ml_trainer import run_wpdp_ml_experiment 
from ui.components.csv_analytics_dialog import CSVAnalyticsDialog

class TrainingWorkerSignals(QObject):
    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

class WPDPTrainingWorker(QRunnable):
    """Worker to run the heavy ML training logic in a separate thread."""
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.signals = TrainingWorkerSignals()

    def run(self):
        try:
            final_results = run_wpdp_ml_experiment(self.config, self.signals.progress.emit)
            self.signals.finished.emit(final_results)
        except Exception as e:
            self.signals.error.emit(f"Experiment Error in Worker: {type(e).__name__}: {str(e)}")


class TestScenarioWV(QWidget):
    """UI for the WPDP Model Training and Evaluation Scenario, including 
    Preprocessing, Feature Selection, and Hyperparameter tuning."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threadpool = QThreadPool()
        self.param_widgets = {} 
        self.init_ui()

    # --- UI INITIALIZATION (init_ui) ---
    def init_ui(self):
        vbox = QVBoxLayout(self)
        vbox.setSpacing(10)
        
        # --- SCROLL AREA SETUP: Contains ALL content ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        config_widget = QWidget() # This widget will contain everything
        config_layout = QVBoxLayout(config_widget)
        
        # --- A. Input/Output Configuration ---
        input_group = QGroupBox("Data & Model Path Configuration")
        input_form = QFormLayout()
        self.data_input = QLineEdit()
        self.browse_data_btn = QPushButton("Browse CSV")
        self.browse_data_btn.clicked.connect(self.select_data_file)
        h_data = QHBoxLayout()
        h_data.addWidget(self.data_input)
        h_data.addWidget(self.browse_data_btn)
        input_form.addRow(QLabel("Input Labeled CSV:"), h_data)
        
        self.save_model_check = QCheckBox("Save Trained Model and Log?")
        self.save_model_check.setChecked(True)
        self.save_model_check.stateChanged.connect(self.toggle_save_model_dir)
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Directory to save model file and log.")
        self.browse_output_btn = QPushButton("Browse Dir")
        self.browse_output_btn.clicked.connect(self.select_output_directory)
        h_output = QHBoxLayout()
        h_output.addWidget(self.output_dir_input)
        h_output.addWidget(self.browse_output_btn)
        input_form.addRow(self.save_model_check)
        input_form.addRow(QLabel("Model/Log Save Dir:"), h_output)
        input_group.setLayout(input_form)
        config_layout.addWidget(input_group) 

        # --- G. Dataset Summary Group (NEW) ---
        self.dataset_summary_group = QGroupBox("Dataset Information")
        self.dataset_summary_group.setVisible(False) # Start hidden
        
        summary_layout = QFormLayout()
        
        # Use QFormLayout for neat two-column display
        self.file_name_label = QLabel("N/A")
        self.file_size_label = QLabel("N/A")
        self.rows_label = QLabel("N/A")
        self.columns_label = QLabel("N/A")
        self.missing_data_label = QLabel("N/A")
        
        summary_layout.addRow(QLabel("<b>File Name:</b>"), self.file_name_label)
        summary_layout.addRow(QLabel("<b>File Size:</b>"), self.file_size_label)
        summary_layout.addRow(QLabel("<b>Rows:</b>"), self.rows_label)
        summary_layout.addRow(QLabel("<b>Columns:</b>"), self.columns_label)
        summary_layout.addRow(QLabel("<b>Missing Data:</b>"), self.missing_data_label)
        
        self.view_analytics_btn = QPushButton("View Detailed Analytics")
        self.view_analytics_btn.clicked.connect(self.view_detailed_analytics)
        self.view_analytics_btn.setStyleSheet("background-color: #2196F3; color: white;")

        vbox_summary_content = QVBoxLayout(self.dataset_summary_group)
        vbox_summary_content.addLayout(summary_layout)
        vbox_summary_content.addWidget(self.view_analytics_btn)
        
        config_layout.addWidget(self.dataset_summary_group)
        
        # --- B. Data Preprocessing Group (REMAINS) ---
        prep_group = QGroupBox("Data Preprocessing")
        # ... (Layout setup B remains unchanged) ...
        prep_grid = QGridLayout()
        self.split_slider = QSlider(Qt.Horizontal)
        self.split_slider.setRange(50, 90)
        self.split_slider.setSingleStep(5)
        self.split_slider.setValue(70)
        self.split_label = QLabel("70% Train / 30% Test")
        self.split_slider.valueChanged.connect(lambda v: self.split_label.setText(f"{v}% Train / {100-v}% Test"))
        prep_grid.addWidget(QLabel("Train/Test Split:"), 0, 0)
        prep_grid.addWidget(self.split_slider, 0, 1)
        prep_grid.addWidget(self.split_label, 0, 2)
        self.norm_check = QCheckBox("Apply Feature Normalization (MinMaxScaler)")
        self.smote_check = QCheckBox("Apply SMOTE (Oversampling)")
        prep_grid.addWidget(self.norm_check, 1, 0, 1, 3)
        prep_grid.addWidget(self.smote_check, 2, 0, 1, 3)
        prep_group.setLayout(prep_grid)
        config_layout.addWidget(prep_group) 
        
        # --- C. Feature Selection Group (REMAINS) ---
        self.fs_group = QGroupBox("Feature Selection")
        # ... (Layout setup C remains unchanged) ...
        fs_grid = QGridLayout()
        self.fs_check = QCheckBox("Apply Feature Selection")
        self.fs_check.stateChanged.connect(self.toggle_feature_selection_options)
        fs_grid.addWidget(self.fs_check, 0, 0, 1, 3)
        self.fs_method_combo = QComboBox()
        self.fs_method_combo.addItems(["SelectKBest (Chi2)", "RFE (Recursive Feature Elimination)", "CSV Filter"])
        self.fs_method_combo.currentIndexChanged.connect(self.update_fs_method_input)
        fs_grid.addWidget(QLabel("Method:"), 1, 0)
        fs_grid.addWidget(self.fs_method_combo, 1, 1, 1, 2)
        self.fs_k_input = QSpinBox()
        self.fs_k_input.setRange(1, 1000)
        self.fs_k_input.setValue(10)
        self.fs_k_input.setVisible(False)
        self.fs_csv_input = QLineEdit()
        self.fs_csv_input.setPlaceholderText("Path to CSV with feature names in the first column.")
        self.fs_csv_input.setVisible(False)
        self.fs_csv_browse_btn = QPushButton("Browse CSV")
        self.fs_csv_browse_btn.clicked.connect(self.select_fs_csv)
        self.fs_csv_browse_btn.setVisible(False)
        self.fs_input_layout = QHBoxLayout()
        self.fs_input_layout.addWidget(self.fs_k_input)
        self.fs_input_layout.addWidget(self.fs_csv_input)
        self.fs_input_layout.addWidget(self.fs_csv_browse_btn)
        fs_grid.addWidget(QLabel("Features (K / CSV):"), 2, 0)
        fs_grid.addLayout(self.fs_input_layout, 2, 1, 1, 2)
        self.fs_group.setLayout(fs_grid)
        self.fs_group.setEnabled(False)
        config_layout.addWidget(self.fs_group) 

        # --- D. Model Selection & Hyperparameters Group (REMAINS) ---
        model_group = QGroupBox("Model Selection & Hyperparameters")
        # ... (Layout setup D remains unchanged) ...
        model_form = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODEL_CONFIGS.keys()))
        self.model_combo.currentIndexChanged.connect(self.update_hyperparams_ui)
        model_form.addRow(QLabel("Select ML Model:"), self.model_combo)
        self.hyperparams_widget = QWidget()
        self.hyperparams_layout = QFormLayout(self.hyperparams_widget)
        self.update_hyperparams_ui() 
        model_form.addRow(QLabel("Hyperparameters:"), self.hyperparams_widget)
        model_group.setLayout(model_form)
        config_layout.addWidget(model_group)

        
        # --- E. Run Button & Progress (MOVED INTO config_layout) ---
        config_layout.addSpacing(15)
        self.train_btn = QPushButton("Start Training and Evaluation")
        self.train_btn.setStyleSheet("background-color: #008CBA; color: white; font-weight: bold; padding: 10px;")
        self.train_btn.clicked.connect(self.start_training)
        config_layout.addWidget(self.train_btn) 
        
        self.progress_bar = QProgressBar()
        config_layout.addWidget(self.progress_bar) 
        
        # --- F. Results Display Group (MOVED INTO config_layout) ---
        self.results_container = QWidget() 
        results_layout_vbox = QVBoxLayout(self.results_container)
        results_group = QGroupBox("Evaluation Results")
        # ... (Layout setup F remains unchanged) ...
        results_layout = QVBoxLayout()
        self.results_summary_label = QLabel("Model: N/A | Split: N/A | Preprocessing: N/A")
        self.results_summary_label.setStyleSheet("font-weight: bold; padding: 5px;")
        self.report_text = QTextEdit("Classification Report will appear here...")
        self.report_text.setReadOnly(True)
        self.report_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.report_text.setFixedHeight(200) 
        self.confusion_matrix_label = QLabel("Confusion Matrix:")
        self.confusion_matrix_display = QLabel("N/A")
        self.confusion_matrix_display.setStyleSheet("white-space: pre;") 
        results_layout.addWidget(self.results_summary_label)
        results_layout.addWidget(QLabel("Classification Report:"))
        results_layout.addWidget(self.report_text)
        results_layout.addWidget(self.confusion_matrix_label)
        results_layout.addWidget(self.confusion_matrix_display)
        results_group.setLayout(results_layout)
        results_layout_vbox.addWidget(results_group)
        
        config_layout.addWidget(self.results_container) 
        self.results_container.setVisible(False) 
        
        config_layout.addStretch(1) # Add stretch to the bottom of the scrollable content

        # Final setup: Add the configured scroll area to the main window layout
        scroll.setWidget(config_widget) 
        vbox.addWidget(scroll)
        
        self.set_ui_state(True)
    
    # --- DATA & ANALYTICS METHODS (NEW / MODIFIED) ---

    def _summarize_dataset_file(self, file_path):
        """Calculates basic metadata for the selected CSV file using standard library and pandas."""
        if not os.path.isfile(file_path):
            return None
        
        summary = {
            'file_name': os.path.basename(file_path),
            'file_size': f"{os.path.getsize(file_path) / (1024 * 1024):.2f} MB",
            'rows': 0,
            'columns': 0,
            'missing_data_status': "N/A",
            'missing_color': "gray"
        }
        
        try:
            # 1. Count rows and columns efficiently using csv module
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                summary['columns'] = len(header)
                summary['rows'] = sum(1 for row in reader) + 1 # +1 for header

            # 2. Check for missing data (using pandas is faster/more robust for this)
            df_temp = pd.read_csv(file_path, nrows=500) # Load small sample for speed
            
            if df_temp.isnull().values.any():
                summary['missing_data_status'] = "Yes (Partial check)"
                summary['missing_color'] = "red"
            else:
                summary['missing_data_status'] = "None (Partial check)"
                summary['missing_color'] = "green"
            
            summary['rows'] = f"{summary['rows']:,}" # Format with comma
            summary['columns'] = f"{summary['columns']}"

            return summary

        except Exception as e:
            QMessageBox.critical(self, "Data Error", f"Error reading file structure:\n{type(e).__name__}: {str(e)}")
            return None
            
    
    def select_data_file(self):
        """Handles file selection and calls the summary update."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Labeled Metrics CSV File", os.getcwd(), "CSV Files (*.csv)")
        if file_path:
            self.data_input.setText(file_path)
            self._update_dataset_summary(file_path)
        else:
            # Clear summary if user cancels selection
            self.dataset_summary_group.setVisible(False)


    def _update_dataset_summary(self, file_path):
        """Updates the UI labels based on the file summary."""
        summary = self._summarize_dataset_file(file_path)
        
        if summary:
            self.file_name_label.setText(summary['file_name'])
            self.file_size_label.setText(summary['file_size'])
            self.rows_label.setText(summary['rows'])
            self.columns_label.setText(summary['columns'])
            
            self.missing_data_label.setText(summary['missing_data_status'])
            self.missing_data_label.setStyleSheet(f"color: {summary['missing_color']};")
            
            self.dataset_summary_group.setVisible(True)
        else:
            self.dataset_summary_group.setVisible(False)
            
    def view_detailed_analytics(self):
        """Opens the CSVAnalyticsDialog component."""
        file_path = self.data_input.text().strip()
        
        if not os.path.isfile(file_path):
            QMessageBox.warning(self, "Data Error", "Please select a valid CSV file before viewing analytics.")
            return

        try:
            # Instantiate and show the external dialog component
            dialog = CSVAnalyticsDialog(file_path, self)
            dialog.exec_()
            
        except NameError:
            # Fallback if CSVAnalyticsDialog is not properly imported/defined
            QMessageBox.critical(self, "Integration Error", 
                                 "The 'CSVAnalyticsDialog' component is missing or not properly imported from 'ui.components.csv_analytics_dialog'.")
        except Exception as e:
            QMessageBox.critical(self, "Analytics Error", f"Failed to load detailed analytics: {str(e)}")

    # --- REMAINDER OF TestScenarioWV METHODS (Unchanged) ---
    # ... (_extract_hyperparameters, start_training, training_finished, 
    #       training_error, set_ui_state, toggle_save_model_dir, 
    #       toggle_feature_selection_options, update_fs_method_input, 
    #       select_output_directory, select_fs_csv, update_progress) 
    #       ... remain here as defined in the user's prompt.
        
    def update_hyperparams_ui(self):
        """Dynamically updates the hyperparameters section based on the selected model config."""
        # ... (implementation as provided by the user) ...
        for i in reversed(range(self.hyperparams_layout.count())): 
            item = self.hyperparams_layout.itemAt(i)
            if item:
                widget_to_remove = item.widget()
                label_to_remove = self.hyperparams_layout.labelForField(widget_to_remove) if widget_to_remove else None
                
                if widget_to_remove:
                    self.hyperparams_layout.removeWidget(widget_to_remove)
                    widget_to_remove.deleteLater()
                if label_to_remove:
                    self.hyperparams_layout.removeWidget(label_to_remove)
                    label_to_remove.deleteLater()
        
        self.param_widgets = {} # Reset the storage for new widgets

        selected_model = self.model_combo.currentText()
        model_data = MODEL_CONFIGS.get(selected_model, {})
        params = model_data.get('params', [])

        for p in params:
            param_name = p['name']
            label = QLabel(f"{param_name} ({p['type']}):")
            input_widget = None

            if p['type'] == 'int':
                widget = QSpinBox()
                widget.setRange(p['range'][0], p['range'][1])
                widget.setValue(int(p['default']))
                input_widget = widget

            elif p['type'] == 'float':
                widget = QDoubleSpinBox()
                widget.setRange(p['range'][0], p['range'][1])
                widget.setSingleStep(p.get('step', 0.1))
                widget.setValue(float(p['default']))
                widget.setDecimals(4) 
                input_widget = widget

            elif p['type'] == 'str':
                widget = QComboBox()
                widget.addItems(p['options'])
                widget.setCurrentText(p['default'])
                input_widget = widget
            
            if input_widget is None:
                input_widget = QLineEdit(str(p['default']))

            self.param_widgets[param_name] = input_widget
            self.hyperparams_layout.addRow(label, input_widget)

    def _extract_hyperparameters(self):
        """Extracts values from the dynamically created parameter widgets."""
        hyperparams = {}
        for name, widget in self.param_widgets.items():
            if isinstance(widget, QSpinBox):
                hyperparams[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                hyperparams[name] = widget.value()
            elif isinstance(widget, QComboBox):
                hyperparams[name] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                try:
                    hyperparams[name] = float(widget.text())
                except ValueError:
                    hyperparams[name] = widget.text()
        return hyperparams

    def start_training(self):
        """Gathers all UI settings and sends them to the worker thread."""
        
        model_hyperparams = self._extract_hyperparameters()
        
        # 1. Gather ALL Configuration
        config = {
            'model': self.model_combo.currentText(),
            'hyperparams': model_hyperparams,
            'data_path': self.data_input.text().strip(),
            'output_dir': self.output_dir_input.text().strip(),
            'save_model': self.save_model_check.isChecked(),
            'split_ratio': self.split_slider.value() / 100.0,
            'normalize': self.norm_check.isChecked(),
            'smote': self.smote_check.isChecked(),
            'fs_apply': self.fs_check.isChecked(),
            'fs_method': self.fs_method_combo.currentText(),
            'fs_k': self.fs_k_input.value() if self.fs_check.isChecked() else None,
            'fs_csv_path': self.fs_csv_input.text().strip() if self.fs_method_combo.currentIndex() == 2 else None,
        }

        # Basic Validation
        if not os.path.isfile(config['data_path']):
            QMessageBox.warning(self, "Input Error", "Please select a valid Labeled Dataset CSV.")
            return
        if config['save_model'] and not config['output_dir']:
            QMessageBox.warning(self, "Input Error", "Please select a valid Model/Log Save Directory.")
            return
        
        self.set_ui_state(False)
        self.progress_bar.setValue(0)
        self.results_container.setVisible(False)
        
        worker = WPDPTrainingWorker(config) 
        
        worker.signals.progress.connect(lambda d: self.update_progress(d, config))
        worker.signals.finished.connect(lambda d: self.training_finished(d, config))
        worker.signals.error.connect(self.training_error)
        
        self.threadpool.start(worker)

    def training_finished(self, metrics_results, config):
        """Processes the detailed results returned by the core logic."""
        self.set_ui_state(True)
        self.progress_bar.setValue(100)
        self.results_container.setVisible(True)
        
        from PyQt5.QtGui import QPixmap 
        from PyQt5.QtCore import Qt 
        
        # 1. Update Summary Label (Unchanged)
        feat_count = metrics_results.get('Feature Count', 'N/A')
        prep_summary = f"Norm: {'Yes' if config['normalize'] else 'No'}, SMOTE: {'Yes' if config['smote'] else 'No'}, FS: {'Yes' if config['fs_apply'] else 'No'} ({feat_count} feats)"
        
        self.results_summary_label.setText(
            f"Model: {config['model']} | Split: {config['split_ratio']:.2f}/{1-config['split_ratio']:.2f} | Preprocessing: {prep_summary}"
        )
        
        # 2. Display Classification Report and Metrics (Unchanged)
        report_text = metrics_results.get('Classification Report Text', "Report text not available from core logic.")
        
        header = f"""
### Model: {config['model']}
Accuracy: {metrics_results.get('Accuracy', 0):.4f} | Precision: {metrics_results.get('Precision', 0):.4f} | Recall: {metrics_results.get('Recall', 0):.4f} | F1: {metrics_results.get('F1 Score', 0):.4f}
Hyperparams: {json.dumps(config['hyperparams'], indent=2)}
---
"""
        self.report_text.setText(header + report_text)
        
        # 3. Display Confusion Matrix (Unchanged)
        cm_plot_path = metrics_results.get('CM Plot Path', 'N/A')
        
        self.confusion_matrix_display.setText("")
        self.confusion_matrix_display.setStyleSheet("white-space: pre;")
        
        if cm_plot_path and os.path.isfile(cm_plot_path):
            self.confusion_matrix_label.setText("Confusion Matrix (Graphical):")
            
            pixmap = QPixmap(cm_plot_path)
            
            self.confusion_matrix_display.setPixmap(pixmap.scaled(
                400, 
                400, 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.confusion_matrix_display.setAlignment(Qt.AlignCenter)
            
        else:
            self.confusion_matrix_label.setText("Confusion Matrix (Text Fallback / Error):")
            cm_data = metrics_results.get('Confusion Matrix', [])
            
            if cm_data and len(cm_data) == 2 and len(cm_data[0]) == 2:
                cm_text = f"""
Predicted: | Bug (1) | Not Bug (0) |
:---: | :---: | :---:
Actual Bug (1): | {cm_data[1][1]} (True Positives) | {cm_data[1][0]} (False Negatives)
Actual Not Bug (0): | {cm_data[0][1]} (False Positives) | {cm_data[0][0]} (True Negatives)
"""
                self.confusion_matrix_display.setText(cm_text)
                self.confusion_matrix_display.setStyleSheet("white-space: pre; font-family: monospace;")
            else:
                error_msg = f"Plot generation failed or file not found at: {cm_plot_path}" if cm_plot_path != 'N/A' else "CM Data N/A"
                self.confusion_matrix_display.setText(error_msg)
                self.confusion_matrix_display.setStyleSheet("color: red; white-space: pre;")

        # 4. Success Message (Unchanged)
        save_msg = f"Model and log saved to: {metrics_results.get('Save Path', 'N/A')}"
        QMessageBox.information(self, "Training Complete", f"Model training and evaluation finished successfully.\n\n{save_msg}")

    def training_error(self, message):
        self.set_ui_state(True)
        self.progress_bar.setValue(0)
        self.results_container.setVisible(True)
        self.results_summary_label.setText("Model: ERROR")
        self.report_text.setText(f"ERROR: {message}")
        self.confusion_matrix_display.setText("N/A")
        QMessageBox.critical(self, "Training Error", message)

    def set_ui_state(self, enabled):
        widgets = [
            self.data_input, self.browse_data_btn,
            self.output_dir_input, self.browse_output_btn,
            self.split_slider, self.norm_check, self.smote_check, 
            self.fs_check, self.fs_method_combo, self.fs_k_input, 
            self.fs_csv_input, self.fs_csv_browse_btn, self.model_combo,
            self.view_analytics_btn # Added new button
        ]
        
        for i in range(self.hyperparams_layout.count()):
            item = self.hyperparams_layout.itemAt(i)
            if item and item.widget():
                widgets.append(item.widget())

        for w in widgets:
            w.setEnabled(enabled)
            
        if self.fs_group.isEnabled() and not enabled:
             self.fs_group.setEnabled(False) 
        elif self.fs_check.isChecked() and enabled:
             self.fs_group.setEnabled(True) 
            
        self.train_btn.setEnabled(enabled)
        if enabled:
            self.train_btn.setText("Start Training and Evaluation")
            self.train_btn.setStyleSheet("background-color: #008CBA; color: white; font-weight: bold; padding: 10px;")
        else:
            self.train_btn.setText("Training in Progress...")
            self.train_btn.setStyleSheet("background-color: #FFC107; color: black; padding: 10px;")

    # --- Placeholder File Dialogs (Retained for completeness) ---
    def toggle_save_model_dir(self, state):
        enabled = state == Qt.Checked
        self.output_dir_input.setEnabled(enabled)
        self.browse_output_btn.setEnabled(enabled)
    def toggle_feature_selection_options(self, state):
        enabled = state == Qt.Checked
        self.fs_group.setEnabled(enabled)
        self.update_fs_method_input(self.fs_method_combo.currentIndex())
    def update_fs_method_input(self, index):
        is_k_based = index in [0, 1]
        is_csv_filter = index == 2
        self.fs_k_input.setVisible(is_k_based and self.fs_check.isChecked())
        self.fs_csv_input.setVisible(is_csv_filter and self.fs_check.isChecked())
        self.fs_csv_browse_btn.setVisible(is_csv_filter and self.fs_check.isChecked())
    def select_output_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Model Output Directory", os.getcwd())
        if dir_path: self.output_dir_input.setText(dir_path)
    def select_fs_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Feature List CSV File", os.getcwd(), "CSV Files (*.csv)")
        if file_path: self.fs_csv_input.setText(file_path)
    def update_progress(self, progress_data, config):
        percent = progress_data.get('percent', 0)
        self.progress_bar.setValue(int(percent))

