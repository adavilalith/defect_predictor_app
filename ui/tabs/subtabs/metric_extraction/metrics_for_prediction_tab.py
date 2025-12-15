import os
import sys 
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLineEdit, QFileDialog, QProgressBar, 
    QMessageBox, QScrollArea, QTableWidget, QTableWidgetItem,
    QSizePolicy, QSpacerItem # Added QSpacerItem for flexible spacing
)
from PyQt5.QtCore import (
    pyqtSignal, QObject, QRunnable, QThreadPool, Qt
)

# Import the core logic (MetricsExtractor no longer needs libclang_path in __init__)
from core.metrics_extractor import MetricsExtractor 
from core.metrics_calculator import MetricsCalculator 

# --- 1. Worker Signals Class (No Change) ---
class WorkerSignals(QObject):
    """Defines the signals available from a running worker thread."""
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    progress = pyqtSignal(float)

# --- 2. Metrics Worker Class (Refactored __init__ and run) ---
class MetricsWorker(QRunnable):
    """
    Worker thread that runs the MetricsExtractor.
    libclang is assumed to be initialized externally.
    """
    # Removed libclang_path from __init__
    def __init__(self, folder_path, output_csv_path):
        super().__init__()
        self.folder_path = folder_path
        self.output_csv_path = output_csv_path
        self.signals = WorkerSignals()
        self.extractor = None

    def run(self):
        """Initialises and runs the long-running process."""
        try:
            # MetricsExtractor is instantiated without the libclang path argument
            self.extractor = MetricsExtractor() 
            
            def progress_reporter(percent):
                self.signals.progress.emit(percent)
                
            df = self.extractor.process_folder(
                self.folder_path,
                self.output_csv_path,
                progress_callback=progress_reporter
            )
            
            self.signals.finished.emit(df)
            
        except Exception as e:
            self.signals.error.emit(str(e))

# ----------------------------------------------------------------------
class MetricsExtractionSubTab(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.threadpool = QThreadPool()
        # libclang initialization is handled by the parent MetricsMainTab
        
        self.df_result = None
        self.init_ui()

    def init_ui(self):
        # --- Widgets ---
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Select C/C++ Source Code Folder...")
        self.browse_folder_btn = QPushButton("Browse Folder")
        
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Select Output CSV File Path...")
        self.browse_output_btn = QPushButton("Browse Output")
        
        self.extract_btn = QPushButton("Extract Metrics and Save")
        self.extract_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        # --- Scrollable Results Table Setup ---
        self.results_table = QTableWidget()
        self.results_table.setRowCount(0)
        self.results_table.setColumnCount(0)
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
        
        # Folder selection layout
        h_folder = QHBoxLayout()
        h_folder.addWidget(self.folder_input)
        h_folder.addWidget(self.browse_folder_btn)
        vbox.addLayout(h_folder)
        
        # Output selection layout
        h_output = QHBoxLayout()
        h_output.addWidget(self.output_input)
        h_output.addWidget(self.browse_output_btn)
        vbox.addLayout(h_output)
        
        vbox.addWidget(self.extract_btn)
        vbox.addWidget(self.progress_bar)
        
        vbox.addWidget(self.scroll_area) 
        
        self.setLayout(vbox)
        
        # --- Signals and Slots ---
        self.browse_folder_btn.clicked.connect(self.select_source_folder)
        self.browse_output_btn.clicked.connect(self.select_output_file)
        self.extract_btn.clicked.connect(self.start_extraction)
        
        self.set_ui_state(True)

    def set_ui_state(self, enabled):
        """Enables/disables UI elements during processing."""
        self.folder_input.setEnabled(enabled)
        self.browse_folder_btn.setEnabled(enabled)
        self.output_input.setEnabled(enabled)
        self.browse_output_btn.setEnabled(enabled)
        self.extract_btn.setEnabled(enabled)
        if enabled:
            self.extract_btn.setText("Extract Metrics and Save")
            self.extract_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        else:
            self.extract_btn.setText("Processing...")
            self.extract_btn.setStyleSheet("background-color: #FFC107; color: black;")

    # --- File Dialog Handlers (No Change) ---
    def select_source_folder(self):
        """Opens a dialog to select the source code folder."""
        folder_path = QFileDialog.getExistingDirectory(
            self, 
            "Select C/C++ Source Code Folder", 
            os.getcwd()
        )
        if folder_path:
            self.folder_input.setText(folder_path)

    def select_output_file(self):
        """Opens a dialog to select the output CSV file path."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Metrics CSV", 
            "metrics_output.csv", 
            "CSV Files (*.csv)"
        )
        if file_path:
            self.output_input.setText(file_path)

    # --- Worker Thread Management (Refactored) ---
    def start_extraction(self):
        """Initiates the metric extraction process in a worker thread."""
        folder_path = self.folder_input.text()
        output_csv_path = self.output_input.text()
        
        # The libclang_path is NO LONGER needed here; it's handled by the parent tab.

        if not folder_path or not os.path.isdir(folder_path):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid source code folder.")
            return

        if not output_csv_path:
            QMessageBox.warning(self, "Invalid Input", "Please select an output CSV file path.")
            return
            
        # 1. Reset UI and set state to disabled
        self.progress_bar.setValue(0)
        self.scroll_area.setVisible(False)
        self.set_ui_state(False)

        # 2. Create the worker (Note: now only passes folder and output paths)
        worker = MetricsWorker(folder_path, output_csv_path)
        worker.signals.progress.connect(self.update_progress)
        worker.signals.finished.connect(self.extraction_finished)
        worker.signals.error.connect(self.extraction_error)
        
        # 3. Execute the worker in the thread pool
        self.threadpool.start(worker)

    # --- Slot Handlers for Worker Signals (No Change) ---
    def update_progress(self, percent):
        """Updates the progress bar and status label (runs on main GUI thread)."""
        self.progress_bar.setValue(int(percent))

    def extraction_finished(self, df):
        """Handles successful completion (runs on main GUI thread)."""
        self.df_result = df
        self.progress_bar.setValue(100)
        self.set_ui_state(True)
        self.preview_results(df)
        
        QMessageBox.information(self, "Success", f"Metrics extraction and saving completed successfully! Saved to {self.output_input.text()}")

    def extraction_error(self, message):
        """Handles errors (runs on main GUI thread)."""
        self.progress_bar.setValue(0)
        self.set_ui_state(True)
        QMessageBox.critical(self, "Extraction Error", f"An error occurred during processing: {message}")

    # --- Result Display (No Change) ---
    def preview_results(self, df):
        """Displays a preview of the resulting DataFrame in the QTableWidget."""
        if df is None or df.empty:
            self.scroll_area.setVisible(False)
            return

        df_preview = df.head(50)
        
        display_columns = [col for col in df_preview.columns]
        
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