import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QFileDialog, 
    QProgressBar, QMessageBox, QScrollArea, QFormLayout, QLabel, QComboBox, 
    QSpinBox, QCheckBox, QGroupBox, QGridLayout, QSlider, QTextEdit, 
    QDoubleSpinBox, QSizePolicy
)
from PyQt5.QtCore import pyqtSignal, QObject, QRunnable, QThreadPool, Qt
from PyQt5.QtGui import QPixmap 
import json 
import pandas as pd 
import csv 

# Assuming these imports are available in the running environment
from core.model_configs import MODEL_CONFIGS 
from core.cvwp_ml_trainer import run_cvwp_ml_experiment # <-- ASSUMING NEW TRAINER
# from ui.components.csv_analytics_dialog import CSVAnalyticsDialog # REMOVED

# UPDATED WORKER CLASS (Uses the new config structure and trainer function)
class TrainingWorkerSignals(QObject):
    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

class CVWPTrainingWorker(QRunnable):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.signals = TrainingWorkerSignals()
    def run(self):
        try:
            # Call the CVWP-specific trainer function
            final_results = run_cvwp_ml_experiment(self.config, self.signals.progress.emit)
            self.signals.finished.emit(final_results)
        except Exception as e:
            self.signals.error.emit(f"Experiment Error in Worker: {type(e).__name__}: {str(e)}")


class TestScenarioCV(QWidget):
    """UI for the CVWP Model Training and Evaluation Scenario, using 
    separate lists of CSVs for Training and Testing."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threadpool = QThreadPool()
        self.param_widgets = {} 
        self.train_files = [] # Stores list of paths for training
        self.test_files = []  # Stores list of paths for testing
        self.init_ui()

    # --------------------------------------------------------------------------
    # 1. init_ui (Cleaned: Removed Dataset Summary Group)
    # --------------------------------------------------------------------------
    def init_ui(self):
        vbox = QVBoxLayout(self)
        vbox.setSpacing(10)
        
        # --- SCROLL AREA SETUP: Contains ALL content ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        config_widget = QWidget() # This widget will contain everything
        config_layout = QVBoxLayout(config_widget)
        
        # A. Input/Output Configuration
        input_group = QGroupBox("Data & Model Path Configuration (CVWP)")
        input_form = QFormLayout()
        
        # --- Training Data List Input ---
        self.train_data_input = QLineEdit()
        self.train_data_input.setPlaceholderText("Select multiple CSVs for training...")
        self.train_data_input.setReadOnly(True) # Display only
        self.browse_train_btn = QPushButton("Select Training CSVs")
        self.browse_train_btn.clicked.connect(self.select_train_data_files)
        h_train_data = QHBoxLayout()
        h_train_data.addWidget(self.train_data_input)
        h_train_data.addWidget(self.browse_train_btn)
        input_form.addRow(QLabel("Input Training CSVs:"), h_train_data)

        # --- Testing Data List Input ---
        self.test_data_input = QLineEdit()
        self.test_data_input.setPlaceholderText("Select multiple CSVs for testing...")
        self.test_data_input.setReadOnly(True) # Display only
        self.browse_test_btn = QPushButton("Select Testing CSVs")
        self.browse_test_btn.clicked.connect(self.select_test_data_files)
        h_test_data = QHBoxLayout()
        h_test_data.addWidget(self.test_data_input)
        h_test_data.addWidget(self.browse_test_btn)
        input_form.addRow(QLabel("Input Testing CSVs:"), h_test_data)
        
        # Save Model/Log Checkbox
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

        # B. Data Preprocessing Group
        prep_group = QGroupBox("Data Preprocessing")
        prep_grid = QGridLayout()
        
        self.norm_check = QCheckBox("Apply Feature Normalization (MinMaxScaler)")
        self.smote_check = QCheckBox("Apply SMOTE (Oversampling on Training Data)")
        prep_grid.addWidget(self.norm_check, 0, 0, 1, 3) 
        prep_grid.addWidget(self.smote_check, 1, 0, 1, 3) 
        prep_group.setLayout(prep_grid)
        config_layout.addWidget(prep_group) 
        
        # C. Feature Selection Group (Unchanged)
        self.fs_group = QGroupBox("Feature Selection")
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
        self.fs_csv_input = QLineEdit()
        self.fs_csv_input.setPlaceholderText("Path to CSV with feature names in the first column.")
        self.fs_csv_browse_btn = QPushButton("Browse CSV")
        self.fs_csv_browse_btn.clicked.connect(self.select_fs_csv)
        self.fs_input_layout = QHBoxLayout()
        self.fs_input_layout.addWidget(self.fs_k_input)
        self.fs_input_layout.addWidget(self.fs_csv_input)
        self.fs_input_layout.addWidget(self.fs_csv_browse_btn)
        fs_grid.addWidget(QLabel("Features (K / CSV):"), 2, 0)
        fs_grid.addLayout(self.fs_input_layout, 2, 1, 1, 2)
        self.fs_group.setLayout(fs_grid)
        config_layout.addWidget(self.fs_group)
        
        # D. Model Selection & Hyperparameters Group (Unchanged)
        model_group = QGroupBox("Model Selection & Hyperparameters")
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

        # E. Run Button & Progress (Unchanged)
        config_layout.addSpacing(15)
        self.train_btn = QPushButton("Start Training and Evaluation")
        self.train_btn.setStyleSheet("background-color: #008CBA; color: white; font-weight: bold; padding: 10px;")
        self.train_btn.clicked.connect(self.start_training)
        config_layout.addWidget(self.train_btn) 
        self.progress_bar = QProgressBar()
        config_layout.addWidget(self.progress_bar) 
        
        # F. Results Display Group (Unchanged)
        self.results_container = QWidget() 
        results_layout_vbox = QVBoxLayout(self.results_container)
        results_group = QGroupBox("Evaluation Results")
        results_layout = QVBoxLayout()
        self.results_summary_label = QLabel("Model: N/A | Dataset: CVWP | Preprocessing: N/A") 
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
        config_layout.addStretch(1)

        scroll.setWidget(config_widget) 
        vbox.addWidget(scroll)
        
        # Final UI state initialization
        self.fs_group.setEnabled(True) 
        self.update_fs_method_input(self.fs_method_combo.currentIndex())
        self.set_ui_state(True)


    # --------------------------------------------------------------------------
    # 2. Data Selection Methods (Cleaned: Removed all summary/analytics calls)
    # --------------------------------------------------------------------------
    
    # NEW: Select multiple files for Training
    def select_train_data_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Training CSV Files (Multiple allowed)", os.getcwd(), "CSV Files (*.csv)"
        )
        if files:
            self.train_files = files
            self.train_data_input.setText(f"{len(files)} files selected.")
        else:
            self.train_files = []
            self.train_data_input.setText("")

    # NEW: Select multiple files for Testing
    def select_test_data_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Testing CSV Files (Multiple allowed)", os.getcwd(), "CSV Files (*.csv)"
        )
        if files:
            self.test_files = files
            self.test_data_input.setText(f"{len(files)} files selected.")
        else:
            self.test_files = []
            self.test_data_input.setText("")

    # REMOVED: view_detailed_analytics, _summarize_dataset_file, _update_dataset_summary

    # The following methods are reused exactly as they were:
    def toggle_save_model_dir(self, state):
        enabled = state == Qt.Checked
        self.output_dir_input.setEnabled(enabled)
        self.browse_output_btn.setEnabled(enabled)

    def toggle_feature_selection_options(self, state):
        self.update_fs_method_input(self.fs_method_combo.currentIndex())

    def update_fs_method_input(self, index):
        is_fs_enabled = self.fs_check.isChecked()
        is_k_based = index in [0, 1]
        is_csv_filter = index == 2
        self.fs_k_input.setVisible(is_k_based and is_fs_enabled)
        self.fs_csv_input.setVisible(is_csv_filter and is_fs_enabled)
        self.fs_csv_browse_btn.setVisible(is_csv_filter and is_fs_enabled)

    def select_output_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Model Output Directory", os.getcwd())
        if dir_path: self.output_dir_input.setText(dir_path)
    
    def select_fs_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Feature List CSV File", os.getcwd(), "CSV Files (*.csv)")
        if file_path: self.fs_csv_input.setText(file_path)

    def update_hyperparams_ui(self):
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
        self.param_widgets = {} 
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
    
    # --------------------------------------------------------------------------
    # 3. start_training (Config Payload Modified)
    # --------------------------------------------------------------------------
    def start_training(self):
        model_hyperparams = self._extract_hyperparameters()
        
        config = {
            'model': self.model_combo.currentText(),
            'hyperparams': model_hyperparams,
            # --- CVWP-SPECIFIC CONFIGURATION ---
            'train_data_paths': self.train_files,      # NEW LIST
            'test_data_paths': self.test_files,        # NEW LIST
            # -----------------------------------
            'output_dir': self.output_dir_input.text().strip(),
            'save_model': self.save_model_check.isChecked(),
            'normalize': self.norm_check.isChecked(),
            'smote': self.smote_check.isChecked(),
            'fs_apply': self.fs_check.isChecked(),
            'fs_method': self.fs_method_combo.currentText(),
            'fs_k': self.fs_k_input.value() if self.fs_check.isChecked() and self.fs_method_combo.currentIndex() in [0, 1] else None,
            'fs_csv_path': self.fs_csv_input.text().strip() if self.fs_method_combo.currentIndex() == 2 and self.fs_check.isChecked() else None,
        }
        
        # --- CVWP-SPECIFIC VALIDATION ---
        if not config['train_data_paths'] or not config['test_data_paths']:
            QMessageBox.warning(self, "Input Error", "Please select valid Training and Testing CSV file lists.")
            return
            
        if config['save_model'] and not config['output_dir']:
            QMessageBox.warning(self, "Input Error", "Please select a valid Model/Log Save Directory.")
            return
            
        self.set_ui_state(False)
        self.progress_bar.setValue(0)
        self.results_container.setVisible(False)
        
        # Use the CVWP-specific worker
        worker = CVWPTrainingWorker(config) 
        worker.signals.progress.connect(lambda d: self.update_progress(d, config))
        worker.signals.finished.connect(lambda d: self.training_finished(d, config))
        worker.signals.error.connect(self.training_error)
        self.threadpool.start(worker)


    # --------------------------------------------------------------------------
    # 4. training_finished (Summary Label Modified)
    # --------------------------------------------------------------------------
    def training_finished(self, metrics_results, config):
        self.set_ui_state(True)
        self.progress_bar.setValue(100)
        self.results_container.setVisible(True)
        feat_count = metrics_results.get('Feature Count', 'N/A')
        
        prep_summary = f"Norm: {'Yes' if config['normalize'] else 'No'}, SMOTE: {'Yes' if config['smote'] else 'No'}, FS: {'Yes' if config['fs_apply'] else 'No'} ({feat_count} feats)"
        
        # MODIFIED: Summary text reflects CVWP nature
        train_count = len(config['train_data_paths'])
        test_count = len(config['test_data_paths'])
        self.results_summary_label.setText(f"Model: {config['model']} | Data: {train_count} Train Files / {test_count} Test Files | Preprocessing: {prep_summary}")
        
        report_text = metrics_results.get('Classification Report Text', "Report text not available from core logic.")
        header = f"""
### Model: {config['model']} (CVWP)
Accuracy: {metrics_results.get('Accuracy', 0):.4f} | Precision: {metrics_results.get('Precision', 0):.4f} | Recall: {metrics_results.get('Recall', 0):.4f} | F1: {metrics_results.get('F1 Score', 0):.4f}
Hyperparams: {json.dumps(config['hyperparams'], indent=2)}
---
"""
        self.report_text.setText(header + report_text)
        
        # Confusion Matrix logic remains the same
        cm_plot_path = metrics_results.get('CM Plot Path', 'N/A')
        self.confusion_matrix_display.setText("")
        self.confusion_matrix_display.setStyleSheet("white-space: pre;")
        if cm_plot_path and os.path.isfile(cm_plot_path):
            self.confusion_matrix_label.setText("Confusion Matrix (Graphical):")
            pixmap = QPixmap(cm_plot_path)
            self.confusion_matrix_display.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
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
        save_msg = f"Model and log saved to: {metrics_results.get('Save Path', 'N/A')}"
        QMessageBox.information(self, "Training Complete", f"Model training and evaluation finished successfully.\n\n{save_msg}")

    # --------------------------------------------------------------------------
    # 5. set_ui_state (Widget List Modified)
    # --------------------------------------------------------------------------
    def set_ui_state(self, enabled):
        """Toggles the enabled state of all configuration widgets."""
        widgets = [
            self.train_data_input, self.browse_train_btn, 
            self.test_data_input, self.browse_test_btn, 
            self.output_dir_input, self.browse_output_btn,
            self.norm_check, self.smote_check, 
            self.fs_method_combo, 
            self.fs_k_input, self.fs_csv_input, self.fs_csv_browse_btn, 
            self.model_combo,
            self.save_model_check,
        ]
        
        # Include dynamic hyperparameters
        for i in range(self.hyperparams_layout.count()):
            item = self.hyperparams_layout.itemAt(i)
            if item and item.widget():
                widgets.append(item.widget())

        for w in widgets:
            w.setEnabled(enabled)
            
        # Feature Selection Group control
        self.fs_group.setEnabled(enabled) 
        
        if enabled:
            self.update_fs_method_input(self.fs_method_combo.currentIndex())
        
        # Run Button State
        self.train_btn.setEnabled(enabled)
        if enabled:
            self.train_btn.setText("Start Training and Evaluation")
            self.train_btn.setStyleSheet("background-color: #008CBA; color: white; font-weight: bold; padding: 10px;")
        else:
            self.train_btn.setText("Training in Progress...")
            self.train_btn.setStyleSheet("background-color: #FFC107; color: black; padding: 10px;")

    def training_error(self, message):
        self.set_ui_state(True)
        self.progress_bar.setValue(0)
        self.results_container.setVisible(True)
        self.results_summary_label.setText("Model: ERROR")
        self.report_text.setText(f"ERROR: {message}")
        self.confusion_matrix_display.setText("N/A")
        QMessageBox.critical(self, "Training Error", message)
    
    def update_progress(self, progress_data, config):
        percent = progress_data.get('percent', 0)
        self.progress_bar.setValue(int(percent))