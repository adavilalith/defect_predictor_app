import os
import json
import pandas as pd
import csv
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QFileDialog, 
    QProgressBar, QMessageBox, QScrollArea, QFormLayout, QLabel, QComboBox, 
    QSpinBox, QCheckBox, QGroupBox, QGridLayout, QSlider, QTextEdit, 
    QDoubleSpinBox, QSizePolicy
)
from PyQt5.QtCore import pyqtSignal, QObject, QRunnable, QThreadPool, Qt
from PyQt5.QtGui import QPixmap 

# Internal Imports
from core.model_configs import MODEL_CONFIGS 
from core.cvwp_ml_trainer import run_cvwp_ml_experiment 
from ui.components.csv_analytics_dialog import CSVAnalyticsDialog

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
            final_results = run_cvwp_ml_experiment(self.config, self.signals.progress.emit)
            self.signals.finished.emit(final_results)
        except Exception as e:
            self.signals.error.emit(f"Experiment Error in Worker: {type(e).__name__}: {str(e)}")

class TestScenarioCV(QWidget):
    """UI for CVWP Model Training with Multiple CSV support and Dataset Inspector."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threadpool = QThreadPool()
        self.param_widgets = {} 
        self.train_files = [] 
        self.test_files = []  
        self.init_ui()

    def init_ui(self):
        vbox = QVBoxLayout(self)
        vbox.setSpacing(10)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        
        # --- A. Input/Output Configuration ---
        input_group = QGroupBox("Data & Model Path Configuration (CVWP)")
        input_form = QFormLayout()
        
        self.train_data_input = QLineEdit()
        self.train_data_input.setPlaceholderText("Select multiple CSVs for training...")
        self.train_data_input.setReadOnly(True)
        self.browse_train_btn = QPushButton("Select Training CSVs")
        self.browse_train_btn.clicked.connect(self.select_train_data_files)
        h_train_data = QHBoxLayout()
        h_train_data.addWidget(self.train_data_input)
        h_train_data.addWidget(self.browse_train_btn)
        input_form.addRow(QLabel("Input Training CSVs:"), h_train_data)

        self.test_data_input = QLineEdit()
        self.test_data_input.setPlaceholderText("Select multiple CSVs for testing...")
        self.test_data_input.setReadOnly(True)
        self.browse_test_btn = QPushButton("Select Testing CSVs")
        self.browse_test_btn.clicked.connect(self.select_test_data_files)
        h_test_data = QHBoxLayout()
        h_test_data.addWidget(self.test_data_input)
        h_test_data.addWidget(self.browse_test_btn)
        input_form.addRow(QLabel("Input Testing CSVs:"), h_test_data)
        
        self.save_model_check = QCheckBox("Save Trained Model and Log?")
        self.save_model_check.setChecked(True)
        self.save_model_check.stateChanged.connect(self.toggle_save_model_dir)
        self.output_dir_input = QLineEdit()
        self.browse_output_btn = QPushButton("Browse Dir")
        self.browse_output_btn.clicked.connect(self.select_output_directory)
        h_output = QHBoxLayout()
        h_output.addWidget(self.output_dir_input)
        h_output.addWidget(self.browse_output_btn)
        input_form.addRow(self.save_model_check)
        input_form.addRow(QLabel("Model/Log Save Dir:"), h_output)
        input_group.setLayout(input_form)
        config_layout.addWidget(input_group) 

        # --- B. Dataset Inspector (NEWLY ADDED) ---
        self.inspector_group = QGroupBox("Dataset Inspector")
        self.inspector_group.setVisible(False)
        inspector_layout = QFormLayout()
        
        self.file_selector_combo = QComboBox()
        self.file_selector_combo.currentIndexChanged.connect(self._on_inspector_file_changed)
        inspector_layout.addRow(QLabel("<b>Inspect File:</b>"), self.file_selector_combo)
        
        self.file_path_label = QLabel("N/A")
        self.file_path_label.setWordWrap(True)
        self.file_size_label = QLabel("N/A")
        self.rows_label = QLabel("N/A")
        self.cols_label = QLabel("N/A")
        
        inspector_layout.addRow(QLabel("Full Path:"), self.file_path_label)
        inspector_layout.addRow(QLabel("File Size:"), self.file_size_label)
        inspector_layout.addRow(QLabel("Rows:"), self.rows_label)
        inspector_layout.addRow(QLabel("Columns:"), self.cols_label)
        
        self.view_analytics_btn = QPushButton("Open Detailed Analytics")
        self.view_analytics_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 5px;")
        self.view_analytics_btn.clicked.connect(self.view_detailed_analytics)
        
        v_ins_content = QVBoxLayout(self.inspector_group)
        v_ins_content.addLayout(inspector_layout)
        v_ins_content.addWidget(self.view_analytics_btn)
        config_layout.addWidget(self.inspector_group)

        # --- C. Data Preprocessing ---
        prep_group = QGroupBox("Data Preprocessing")
        prep_grid = QGridLayout()
        self.norm_check = QCheckBox("Apply Feature Normalization (MinMaxScaler)")
        self.smote_check = QCheckBox("Apply SMOTE (Oversampling on Training Data)")
        prep_grid.addWidget(self.norm_check, 0, 0, 1, 3) 
        prep_grid.addWidget(self.smote_check, 1, 0, 1, 3) 
        prep_group.setLayout(prep_grid)
        config_layout.addWidget(prep_group) 
        
        # --- D. Feature Selection ---
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
        
        # --- E. Model Selection ---
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

        # --- F. Run & Progress ---
        config_layout.addSpacing(15)
        self.train_btn = QPushButton("Start Training and Evaluation")
        self.train_btn.setStyleSheet("background-color: #008CBA; color: white; font-weight: bold; padding: 10px;")
        self.train_btn.clicked.connect(self.start_training)
        config_layout.addWidget(self.train_btn) 
        self.progress_bar = QProgressBar()
        config_layout.addWidget(self.progress_bar) 
        
        # --- G. Results Display ---
        self.results_container = QWidget() 
        results_layout_vbox = QVBoxLayout(self.results_container)
        results_group = QGroupBox("Evaluation Results")
        results_layout = QVBoxLayout()
        self.results_summary_label = QLabel("Model: N/A | Dataset: CVWP | Preprocessing: N/A") 
        self.results_summary_label.setStyleSheet("font-weight: bold; padding: 5px;")
        self.report_text = QTextEdit("Classification Report will appear here...")
        self.report_text.setReadOnly(True)
        self.report_text.setFixedHeight(200) 
        self.confusion_matrix_label = QLabel("Confusion Matrix:")
        self.confusion_matrix_display = QLabel("N/A")
        results_layout.addWidget(self.results_summary_label)
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
        
        self.fs_group.setEnabled(True) 
        self.update_fs_method_input(self.fs_method_combo.currentIndex())
        self.set_ui_state(True)

    # --- Inspector & Data Selection Methods ---
    def _refresh_inspector_list(self):
        self.file_selector_combo.blockSignals(True)
        self.file_selector_combo.clear()
        all_files = self.train_files + self.test_files
        if not all_files:
            self.inspector_group.setVisible(False)
            self.file_selector_combo.blockSignals(False)
            return
        for path in all_files:
            prefix = "[Train]" if path in self.train_files else "[Test]"
            self.file_selector_combo.addItem(f"{prefix} {os.path.basename(path)}", path)
        self.inspector_group.setVisible(True)
        self.file_selector_combo.blockSignals(False)
        self._on_inspector_file_changed(0)

    def _on_inspector_file_changed(self, index):
        file_path = self.file_selector_combo.currentData()
        if not file_path or not os.path.exists(file_path): return
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            df_temp = pd.read_csv(file_path, nrows=0)
            with open(file_path, 'r', encoding='utf-8') as f:
                row_count = sum(1 for _ in f) - 1
            self.file_path_label.setText(file_path)
            self.file_size_label.setText(f"{size_mb:.2f} MB")
            self.rows_label.setText(f"{row_count:,}")
            self.cols_label.setText(str(len(df_temp.columns)))
        except Exception as e:
            self.file_path_label.setText(f"Error: {str(e)}")

    def view_detailed_analytics(self):
        file_path = self.file_selector_combo.currentData()
        if file_path:
            dialog = CSVAnalyticsDialog(file_path, self)
            dialog.exec_()

    def select_train_data_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Training CSVs", os.getcwd(), "CSV Files (*.csv)")
        if files:
            self.train_files = files
            self.train_data_input.setText(f"{len(files)} files selected.")
            self._refresh_inspector_list()

    def select_test_data_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Testing CSVs", os.getcwd(), "CSV Files (*.csv)")
        if files:
            self.test_files = files
            self.test_data_input.setText(f"{len(files)} files selected.")
            self._refresh_inspector_list()

    # --- Shared Utility Methods ---
    def toggle_save_model_dir(self, state):
        enabled = state == Qt.Checked
        self.output_dir_input.setEnabled(enabled)
        self.browse_output_btn.setEnabled(enabled)

    def toggle_feature_selection_options(self, state):
        self.update_fs_method_input(self.fs_method_combo.currentIndex())

    def update_fs_method_input(self, index):
        is_fs = self.fs_check.isChecked()
        self.fs_k_input.setVisible(index in [0, 1] and is_fs)
        self.fs_csv_input.setVisible(index == 2 and is_fs)
        self.fs_csv_browse_btn.setVisible(index == 2 and is_fs)

    def select_output_directory(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Dir")
        if path: self.output_dir_input.setText(path)
    
    def select_fs_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Feature List", filter="CSV (*.csv)")
        if path: self.fs_csv_input.setText(path)

    def update_hyperparams_ui(self):
        for i in reversed(range(self.hyperparams_layout.count())):
            item = self.hyperparams_layout.itemAt(i)
            if item and item.widget(): item.widget().deleteLater()
        self.param_widgets = {}
        model_data = MODEL_CONFIGS.get(self.model_combo.currentText(), {})
        for p in model_data.get('params', []):
            label = QLabel(f"{p['name']}:")
            if p['type'] == 'int':
                w = QSpinBox(); w.setRange(*p['range']); w.setValue(p['default'])
            elif p['type'] == 'float':
                w = QDoubleSpinBox(); w.setRange(*p['range']); w.setValue(p['default']); w.setDecimals(4)
            elif p['type'] == 'str':
                w = QComboBox(); w.addItems(p['options']); w.setCurrentText(p['default'])
            else: w = QLineEdit(str(p['default']))
            self.param_widgets[p['name']] = w
            self.hyperparams_layout.addRow(label, w)

    def _extract_hyperparameters(self):
        hp = {}
        for n, w in self.param_widgets.items():
            if isinstance(w, QSpinBox) or isinstance(w, QDoubleSpinBox): hp[n] = w.value()
            elif isinstance(w, QComboBox): hp[n] = w.currentText()
            else: hp[n] = w.text()
        return hp

    # --- Execution Logic ---
    def start_training(self):
        config = {
            'model': self.model_combo.currentText(),
            'hyperparams': self._extract_hyperparameters(),
            'train_data_paths': self.train_files,
            'test_data_paths': self.test_files,
            'output_dir': self.output_dir_input.text().strip(),
            'save_model': self.save_model_check.isChecked(),
            'normalize': self.norm_check.isChecked(),
            'smote': self.smote_check.isChecked(),
            'fs_apply': self.fs_check.isChecked(),
            'fs_method': self.fs_method_combo.currentText(),
            'fs_k': self.fs_k_input.value() if self.fs_check.isChecked() and self.fs_method_combo.currentIndex() < 2 else None,
            'fs_csv_path': self.fs_csv_input.text().strip() if self.fs_check.isChecked() and self.fs_method_combo.currentIndex() == 2 else None,
        }
        if not self.train_files or not self.test_files:
            QMessageBox.warning(self, "Error", "Missing training or testing data.")
            return
        self.set_ui_state(False)
        worker = CVWPTrainingWorker(config) 
        worker.signals.progress.connect(lambda d: self.progress_bar.setValue(int(d.get('percent', 0))))
        worker.signals.finished.connect(lambda d: self.training_finished(d, config))
        worker.signals.error.connect(self.training_error)
        self.threadpool.start(worker)

    def training_finished(self, res, cfg):
        self.set_ui_state(True)
        self.results_container.setVisible(True)
        summary = f"Model: {cfg['model']} | {len(cfg['train_data_paths'])} Train / {len(cfg['test_data_paths'])} Test"
        self.results_summary_label.setText(summary)
        self.report_text.setText(res.get('Classification Report Text', 'No report'))
        
        path = res.get('CM Plot Path')
        if path and os.path.exists(path):
            self.confusion_matrix_display.setPixmap(QPixmap(path).scaled(400, 400, Qt.KeepAspectRatio))
        else:
            self.confusion_matrix_display.setText(str(res.get('Confusion Matrix', 'N/A')))
        QMessageBox.information(self, "Success", "Training Complete")

    def training_error(self, msg):
        self.set_ui_state(True)
        QMessageBox.critical(self, "Error", msg)

    def set_ui_state(self, state):
        for w in [self.browse_train_btn, self.browse_test_btn, self.train_btn, self.fs_group, self.model_combo]:
            w.setEnabled(state)