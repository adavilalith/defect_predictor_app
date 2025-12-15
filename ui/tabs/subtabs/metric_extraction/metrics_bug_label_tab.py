import os
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLineEdit, QFileDialog, QProgressBar, 
    QMessageBox, QScrollArea, QTableWidget, QTableWidgetItem,
    QSizePolicy
)
from PyQt5.QtCore import (
    pyqtSignal, QObject, QRunnable, QThreadPool, Qt
)

# Import the new end-to-end logic
from core.extract_add_bug import extract_metrics_and_add_bug_label 

# --- 1. Worker Signals Class (Reused) ---
class WorkerSignals(QObject):
    """Defines the signals available from a running worker thread."""
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    progress = pyqtSignal(float)


# --- 2. Data Labeling Worker Class (Updated to use the core function) ---
class BugLabelingWorker(QRunnable):
    """
    Worker thread to delegate the entire data preparation process 
    to the core extract_metrics_and_add_bug_label function.
    """
    def __init__(self, source_folder, bug_report_csv, output_csv, bug_function_name):
        super().__init__()
        self.source_folder = source_folder
        self.bug_report_csv = bug_report_csv
        self.output_csv = output_csv
        self.bug_function_name = bug_function_name
        self.signals = WorkerSignals()

    def run(self):
        """Runs the extraction and labeling process."""
        try:
            # Delegate all complex steps to the core function
            df_metrics = extract_metrics_and_add_bug_label(
                source_folder=self.source_folder,
                bug_report_csv=self.bug_report_csv,
                output_csv=self.output_csv,
                bug_function_name_col=self.bug_function_name,
                progress_callback=lambda p: self.signals.progress.emit(p)
            )
            
            self.signals.finished.emit(df_metrics)
            
        except Exception as e:
            # Catch the exception raised by the core function
            self.signals.error.emit(f"Data Preparation Error: {str(e)}")


# --- 3. Data Preparation UI Tab ---
class MetricsBugLabelSubTab(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.threadpool = QThreadPool()
        self.df_result = None
        self.init_ui()

    def init_ui(self):
        # --- Widgets ---
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("1. Select C/C++ Source Code Folder...")
        self.browse_folder_btn = QPushButton("Source Folder")
        
        self.bug_report_input = QLineEdit()
        self.bug_report_input.setPlaceholderText("2. Select Bug Report CSV File (e.g., function list)...")
        self.browse_bug_btn = QPushButton("Bug CSV")
        
        self.bug_function_name_input = QLineEdit()
        self.bug_function_name_input.setPlaceholderText("3. Enter Function Name Column in Bug CSV (optional)")
        
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("4. Select Output Labeled CSV File Path...")
        self.browse_output_btn = QPushButton("Save Output")
        
        self.label_btn = QPushButton("Run Data Preparation (Extract & Label)")
        self.label_btn.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        # --- Scrollable Results Table Setup ---
        self.results_table = QTableWidget()
        self.results_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.results_table.setSizePolicy(
            QSizePolicy.Expanding, 
            QSizePolicy.Expanding
        )
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.results_table)
        self.scroll_area.setVisible(False)
        
        # --- Layouts ---
        vbox = QVBoxLayout()
        vbox.setSpacing(10)
        
        # 1. Source Folder
        h_folder = QHBoxLayout()
        h_folder.addWidget(self.folder_input)
        h_folder.addWidget(self.browse_folder_btn)
        vbox.addLayout(h_folder)
        
        # 2. Bug Report CSV
        h_bug = QHBoxLayout()
        h_bug.addWidget(self.bug_report_input)
        h_bug.addWidget(self.browse_bug_btn)
        vbox.addLayout(h_bug)
        
        # 3. Bug Function Name Column
        vbox.addWidget(self.bug_function_name_input)
        
        # 4. Output Selection
        h_output = QHBoxLayout()
        h_output.addWidget(self.output_input)
        h_output.addWidget(self.browse_output_btn)
        vbox.addLayout(h_output)
        
        vbox.addWidget(self.label_btn)
        vbox.addWidget(self.progress_bar)
        
        # The scroll area fills all remaining space
        vbox.addWidget(self.scroll_area) 

        # Add a stretch at the end to ensure all content is pushed to the top when the scroll area is hidden
        vbox.addStretch(1) 

        self.setLayout(vbox)
        
        # --- Signals and Slots ---
        self.browse_folder_btn.clicked.connect(self.select_source_folder)
        self.browse_bug_btn.clicked.connect(self.select_bug_report_file)
        self.browse_output_btn.clicked.connect(self.select_output_file)
        self.label_btn.clicked.connect(self.start_labeling)
        
        self.set_ui_state(True)
        
    def set_ui_state(self, enabled):
        """Enables/disables UI elements during processing."""
        self.folder_input.setEnabled(enabled)
        self.browse_folder_btn.setEnabled(enabled)
        self.bug_report_input.setEnabled(enabled)
        self.browse_bug_btn.setEnabled(enabled)
        self.bug_function_name_input.setEnabled(enabled)
        self.output_input.setEnabled(enabled)
        self.browse_output_btn.setEnabled(enabled)
        self.label_btn.setEnabled(enabled)
        if enabled:
            self.label_btn.setText("Run Data Preparation (Extract & Label)")
            self.label_btn.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold;")
        else:
            self.label_btn.setText("Processing...")
            self.label_btn.setStyleSheet("background-color: #FFC107; color: black;")

    # --- File Dialog Handlers ---
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

    # --- Worker Thread Management ---
    def start_labeling(self):
        """Initiates the data preparation process in a worker thread."""
        source_folder = self.folder_input.text()
        bug_report_csv = self.bug_report_input.text()
        output_csv = self.output_input.text()
        bug_function_name = self.bug_function_name_input.text()

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

    # --- Slot Handlers for Worker Signals ---
    def update_progress(self, percent):
        """Updates the progress bar (runs on main GUI thread)."""
        self.progress_bar.setValue(int(percent))

    def labeling_finished(self, df):
        """Handles successful completion (runs on main GUI thread)."""
        self.df_result = df
        self.progress_bar.setValue(100)
        self.set_ui_state(True)
        self.preview_results(df)
        
        QMessageBox.information(self, "Success", f"Data preparation and saving completed successfully! Saved to {self.output_input.text()}")

    def labeling_error(self, message):
        """Handles errors (runs on main GUI thread)."""
        self.progress_bar.setValue(0)
        self.set_ui_state(True)
        QMessageBox.critical(self, "Data Preparation Error", message)

    # --- Result Display ---
    def preview_results(self, df):
        """Displays a preview of the resulting DataFrame in the QTableWidget."""
        if df is None or df.empty:
            self.scroll_area.setVisible(False)
            return

        # Show the first 50 rows, as before
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