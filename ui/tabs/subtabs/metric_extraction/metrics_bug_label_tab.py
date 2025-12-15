import os
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLineEdit, QFileDialog, QProgressBar, 
    QMessageBox, QScrollArea, QTableWidget, QTableWidgetItem,
    QSizePolicy, QFormLayout, QLabel
)
from PyQt5.QtCore import (
    pyqtSignal, QObject, QRunnable, QThreadPool, Qt
)

# NOTE: The imports below are assumed to exist in your environment.
from core.extract_add_bug import extract_metrics_and_add_bug_label 

from ui.components.csv_analytics_dialog import CSVAnalyticsDialog

# --- Worker Signals and Worker Classes (Unchanged from previous response) ---
class WorkerSignals(QObject):
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    progress = pyqtSignal(float)

class BugLabelingWorker(QRunnable):
    def __init__(self, source_folder, bug_report_csv, output_csv, bug_function_name):
        super().__init__()
        self.source_folder = source_folder
        self.bug_report_csv = bug_report_csv
        self.output_csv = output_csv
        self.bug_function_name = bug_function_name
        self.signals = WorkerSignals()

    def run(self):
        try:
            df_metrics = extract_metrics_and_add_bug_label(
                source_folder=self.source_folder,
                bug_report_csv=self.bug_report_csv,
                output_csv=self.output_csv,
                bug_function_name_col=self.bug_function_name,
                progress_callback=lambda p: self.signals.progress.emit(p)
            )
            self.signals.finished.emit(df_metrics)
        except Exception as e:
            self.signals.error.emit(f"Data Preparation Error: {str(e)}")


# --- 3. Data Preparation UI Tab (UPDATED) ---
class MetricsBugLabelSubTab(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threadpool = QThreadPool()
        self.df_result = None
        self.init_ui()

    def init_ui(self):
        vbox = QVBoxLayout(self)
        vbox.setSpacing(10)
        
        # Style for the Analytics Button
        analytics_btn_style = """
            QPushButton {
                background-color: #2196F3; 
                color: white; 
                padding: 4px; 
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """

        input_form = QFormLayout()

        # 1. Source Folder Input (No Analytics button needed here, as it's a folder)
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("e.g., /project/src")
        self.browse_folder_btn = QPushButton("Browse Folder")
        self.browse_folder_btn.clicked.connect(self.select_source_folder)
        
        h_folder = QHBoxLayout()
        h_folder.addWidget(self.folder_input)
        h_folder.addWidget(self.browse_folder_btn)
        input_form.addRow(QLabel("Source Code Folder:"), h_folder)
        
        # 2. Bug Report CSV Input (ANALYTICS BUTTON ADDED)
        self.bug_report_input = QLineEdit()
        self.bug_report_input.setPlaceholderText("e.g., /data/bugs.csv")
        self.browse_bug_btn = QPushButton("Browse CSV")
        self.browse_bug_btn.clicked.connect(self.select_bug_report_file)
        
        analytics_btn_bug = QPushButton("View Analytics")
        analytics_btn_bug.setStyleSheet(analytics_btn_style)
        # Connect to the new handler method
        analytics_btn_bug.clicked.connect(lambda: self.prompt_show_analytics(self.bug_report_input))
        
        h_bug = QHBoxLayout()
        h_bug.addWidget(self.bug_report_input)
        h_bug.addWidget(self.browse_bug_btn)
        h_bug.addWidget(analytics_btn_bug) # Add the analytics button
        input_form.addRow(QLabel("Bug Report CSV File:"), h_bug)
        
        # 3. Bug Function Name Column
        self.bug_function_name_input = QLineEdit()
        self.bug_function_name_input.setPlaceholderText("e.g., 'Function_Name' (Column in Bug CSV)")
        input_form.addRow(QLabel("Bug Name Column:"), self.bug_function_name_input)
        
        # 4. Output Selection (ANALYTICS BUTTON ADDED)
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("e.g., /output/labeled_metrics.csv")
        self.browse_output_btn = QPushButton("Save Output")
        self.browse_output_btn.clicked.connect(self.select_output_file)
        
        analytics_btn_output = QPushButton("View Analytics")
        analytics_btn_output.setStyleSheet(analytics_btn_style)
        # Connect to the new handler method
        analytics_btn_output.clicked.connect(lambda: self.prompt_show_analytics(self.output_input))
        
        h_output = QHBoxLayout()
        h_output.addWidget(self.output_input)
        h_output.addWidget(self.browse_output_btn)
        h_output.addWidget(analytics_btn_output) # Add the analytics button
        input_form.addRow(QLabel("Output Labeled CSV:"), h_output)

        vbox.addLayout(input_form)
        
        # --- Run Button & Progress (Unchanged) ---
        self.label_btn = QPushButton("Run Data Preparation (Extract Metrics & Add Bug Label)")
        self.label_btn.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 10px;")
        self.label_btn.clicked.connect(self.start_labeling)
        vbox.addWidget(self.label_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        vbox.addWidget(self.progress_bar)
        
        # --- Scrollable Results Table Setup (Unchanged) ---
        self.results_table = QTableWidget()
        self.results_table.setWordWrap(False)
        self.results_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.results_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.results_table)
        self.scroll_area.setVisible(False)
        
        vbox.addWidget(self.scroll_area) 
        vbox.addStretch(1) 

        self.set_ui_state(True)
        
    # --- New Analytics Helper Method ---


    def prompt_show_analytics(self, line_edit_widget: QLineEdit):
        """
        Handles the click for 'View Analytics' button. 
        It checks the path and then calls a method (which should launch a dialog).
        """
        csv_path = line_edit_widget.text().strip()
        if csv_path and os.path.exists(csv_path):
            analytics_dialog = CSVAnalyticsDialog(csv_path, self)
            analytics_dialog.exec_()
        else:
            QMessageBox.warning(self, "Warning", "Please select a valid CSV file first.")

    # --- Other Methods (Unchanged: set_ui_state, select_*, start_labeling, update_progress, etc.) ---
    def set_ui_state(self, enabled):
        """Enables/disables UI elements during processing."""
        widgets = [
            self.folder_input, self.browse_folder_btn,
            self.bug_report_input, self.browse_bug_btn,
            self.bug_function_name_input,
            self.output_input, self.browse_output_btn
        ]
        for w in widgets:
            w.setEnabled(enabled)
            
        self.label_btn.setEnabled(enabled)
        if enabled:
            self.label_btn.setText("Run Data Preparation (Extract Metrics & Add Bug Label)")
            self.label_btn.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 10px;")
        else:
            self.label_btn.setText("Processing... Please wait for extraction and labeling to complete.")
            self.label_btn.setStyleSheet("background-color: #FFC107; color: black; padding: 10px;")

    def select_source_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select C/C++ Source Code Folder", os.getcwd()
        )
        if folder_path:
            self.folder_input.setText(folder_path)

    def select_bug_report_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Bug Report CSV File", os.getcwd(), "CSV Files (*.csv)"
        )
        if file_path:
            self.bug_report_input.setText(file_path)

    def select_output_file(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Labeled Dataset CSV", "labeled_dataset.csv", "CSV Files (*.csv)"
        )
        if file_path:
            self.output_input.setText(file_path)

    def start_labeling(self):
        source_folder = self.folder_input.text().strip()
        bug_report_csv = self.bug_report_input.text().strip()
        output_csv = self.output_input.text().strip()
        bug_function_name = self.bug_function_name_input.text().strip()
        
        if not source_folder or not os.path.isdir(source_folder):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid source code folder.")
            return

        if not bug_report_csv or not os.path.isfile(bug_report_csv):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid bug report CSV file.")
            return

        if not output_csv:
            QMessageBox.warning(self, "Invalid Input", "Please select an output CSV file path.")
            return
            
        self.progress_bar.setValue(0)
        self.scroll_area.setVisible(False)
        self.set_ui_state(False)

        worker = BugLabelingWorker(source_folder, bug_report_csv, output_csv, bug_function_name)
        worker.signals.progress.connect(self.update_progress)
        worker.signals.finished.connect(self.labeling_finished)
        worker.signals.error.connect(self.labeling_error)
        
        self.threadpool.start(worker)

    def update_progress(self, percent):
        self.progress_bar.setValue(int(percent))

    def labeling_finished(self, df):
        self.df_result = df
        self.progress_bar.setValue(100)
        self.set_ui_state(True)
        self.preview_results(df)
        
        QMessageBox.information(
            self, 
            "Success", 
            f"Data preparation and saving completed successfully! Saved to:\n{self.output_input.text()}"
        )

    def labeling_error(self, message):
        self.progress_bar.setValue(0)
        self.set_ui_state(True)
        QMessageBox.critical(self, "Data Preparation Error", message)

    def preview_results(self, df):
        if df is None or df.empty:
            self.scroll_area.setVisible(False)
            return

        df_preview = df.head(50)
        display_columns = list(df_preview.columns)
        
        self.results_table.setColumnCount(len(display_columns))
        self.results_table.setHorizontalHeaderLabels(display_columns)
        self.results_table.setRowCount(len(df_preview))
        
        for i, row in df_preview.iterrows():
            for j, col in enumerate(display_columns):
                value = row[col]
                if isinstance(value, (float,)):
                    display_text = f"{value:.2f}"
                else:
                    display_text = str(value)

                item = QTableWidgetItem(display_text)
                item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(i, j, item)
        
        self.results_table.resizeColumnsToContents()
        self.scroll_area.setVisible(True)