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
# from core.extract_add_bug import extract_metrics_and_add_bug_label 
# We'll need a different core function for this tab, let's call it:
from core.add_bug_label_to_existing_metrics import process_existing_metrics_and_add_bug

from ui.components.csv_analytics_dialog import CSVAnalyticsDialog


# --- Worker Signals Class (Reused) ---
class WorkerSignals(QObject):
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    progress = pyqtSignal(float)

# --- New Worker Class for Existing Metrics ---
class ExistingMetricsWorker(QRunnable):
    """
    Worker thread for the "Use Existing Metrics" tab.
    """
    def __init__(self, existing_metrics_csv, bug_report_csv, output_csv, bug_function_name):
        super().__init__()
        self.existing_metrics_csv = existing_metrics_csv
        self.bug_report_csv = bug_report_csv
        self.output_csv = output_csv
        self.bug_function_name = bug_function_name
        self.signals = WorkerSignals()

    def run(self):
        try:
            df_labeled = process_existing_metrics_and_add_bug(
                metrics_csv_path=self.existing_metrics_csv,
                bug_report_csv=self.bug_report_csv,
                output_csv=self.output_csv,
                bug_function_name_col=self.bug_function_name,
                progress_callback=lambda p: self.signals.progress.emit(p)
            )
            self.signals.finished.emit(df_labeled)
        except Exception as e:
            self.signals.error.emit(f"Data Preparation Error: {str(e)}")


# --- 4. Use Existing Metrics UI Tab ---
class UseExistingMetricsSubTab(QWidget):
    """
    UI Tab for using existing metrics CSV and merging bug labels.
    Corresponds to Tab Index 3.
    """
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

        # 1. Existing Metrics CSV Input (NEW INPUT, with Analytics)
        self.existing_metrics_input = QLineEdit()
        self.existing_metrics_input.setPlaceholderText("e.g., /output/metrics_only.csv")
        self.browse_metrics_btn = QPushButton("Browse CSV")
        self.browse_metrics_btn.clicked.connect(self.select_existing_metrics_file)
        
        analytics_btn_metrics = QPushButton("View Analytics")
        analytics_btn_metrics.setStyleSheet(analytics_btn_style)
        analytics_btn_metrics.clicked.connect(lambda: self.prompt_show_analytics(self.existing_metrics_input))
        
        h_metrics = QHBoxLayout()
        h_metrics.addWidget(self.existing_metrics_input)
        h_metrics.addWidget(self.browse_metrics_btn)
        h_metrics.addWidget(analytics_btn_metrics)
        input_form.addRow(QLabel("Existing Metrics CSV:"), h_metrics)
        
        # 2. Bug Report CSV Input (Reused from previous tab)
        self.bug_report_input = QLineEdit()
        self.bug_report_input.setPlaceholderText("e.g., /data/bugs.csv")
        self.browse_bug_btn = QPushButton("Browse CSV")
        self.browse_bug_btn.clicked.connect(self.select_bug_report_file)
        
        analytics_btn_bug = QPushButton("View Analytics")
        analytics_btn_bug.setStyleSheet(analytics_btn_style)
        analytics_btn_bug.clicked.connect(lambda: self.prompt_show_analytics(self.bug_report_input))
        
        h_bug = QHBoxLayout()
        h_bug.addWidget(self.bug_report_input)
        h_bug.addWidget(self.browse_bug_btn)
        h_bug.addWidget(analytics_btn_bug)
        input_form.addRow(QLabel("Bug Report CSV File:"), h_bug)
        
        # 3. Bug Function Name Column (Reused)
        self.bug_function_name_input = QLineEdit()
        self.bug_function_name_input.setPlaceholderText("e.g., 'Function_Name' (Column in Bug CSV)")
        input_form.addRow(QLabel("Bug Name Column:"), self.bug_function_name_input)
        
        # 4. Output Selection (Reused, with Analytics)
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("e.g., /output/labeled_dataset.csv")
        self.browse_output_btn = QPushButton("Save Output")
        self.browse_output_btn.clicked.connect(self.select_output_file)
        
        analytics_btn_output = QPushButton("View Analytics")
        analytics_btn_output.setStyleSheet(analytics_btn_style)
        analytics_btn_output.clicked.connect(lambda: self.prompt_show_analytics(self.output_input))
        
        h_output = QHBoxLayout()
        h_output.addWidget(self.output_input)
        h_output.addWidget(self.browse_output_btn)
        h_output.addWidget(analytics_btn_output)
        input_form.addRow(QLabel("Output Labeled CSV:"), h_output)

        vbox.addLayout(input_form)
        
        # --- Run Button & Progress ---
        self.label_btn = QPushButton("Run Labeling (Using Existing Metrics)")
        self.label_btn.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 10px;")
        self.label_btn.clicked.connect(self.start_labeling)
        vbox.addWidget(self.label_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        vbox.addWidget(self.progress_bar)
        
        # --- Scrollable Results Table Setup ---
        self.results_table = QTableWidget()
        self.results_table.setWordWrap(False)
        self.results_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.results_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.results_table)
        self.scroll_area.setVisible(False)
        
        vbox.addWidget(self.scroll_area) 
        vbox.addStretch(0) 

        self.set_ui_state(True)
        
    # --- File Dialog Handlers ---
    def select_existing_metrics_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Existing Metrics CSV File", os.getcwd(), "CSV Files (*.csv)"
        )
        if file_path:
            self.existing_metrics_input.setText(file_path)

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

    # --- Analytics and Worker Thread Management ---
    def prompt_show_analytics(self, line_edit_widget: QLineEdit):
        """
        Handles the click for 'View Analytics' button, launching the CSVAnalyticsDialog.
        """
        csv_path = line_edit_widget.text().strip()
        if csv_path and os.path.exists(csv_path):
            analytics_dialog = CSVAnalyticsDialog(csv_path, self)
            analytics_dialog.exec_()
        else:
            QMessageBox.warning(self, "Warning", "Please select a valid CSV file first.")

    def start_labeling(self):
        """Initiates the data preparation process in a worker thread."""
        existing_metrics_csv = self.existing_metrics_input.text().strip()
        bug_report_csv = self.bug_report_input.text().strip()
        output_csv = self.output_input.text().strip()
        bug_function_name = self.bug_function_name_input.text().strip()
        
        # --- Validation ---
        if not existing_metrics_csv or not os.path.isfile(existing_metrics_csv):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid existing metrics CSV file.")
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

        # Use the new ExistingMetricsWorker
        worker = ExistingMetricsWorker(existing_metrics_csv, bug_report_csv, output_csv, bug_function_name)
        worker.signals.progress.connect(self.update_progress)
        worker.signals.finished.connect(self.labeling_finished)
        worker.signals.error.connect(self.labeling_error)
        
        self.threadpool.start(worker)

    # --- Slot Handlers (Unchanged) ---
    def set_ui_state(self, enabled):
        """Enables/disables UI elements during processing."""
        widgets = [
            self.existing_metrics_input, self.browse_metrics_btn,
            self.bug_report_input, self.browse_bug_btn,
            self.bug_function_name_input,
            self.output_input, self.browse_output_btn
        ]
        for w in widgets:
            w.setEnabled(enabled)
            
        self.label_btn.setEnabled(enabled)
        if enabled:
            self.label_btn.setText("Run Labeling (Using Existing Metrics)")
            self.label_btn.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 10px;")
        else:
            self.label_btn.setText("Processing... Please wait for labeling to complete.")
            self.label_btn.setStyleSheet("background-color: #FFC107; color: black; padding: 10px;")

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
            f"Labeling and saving completed successfully! Saved to:\n{self.output_input.text()}"
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

# --- Example Execution Block (Optional, for testing) ---
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication, QTabWidget
    import sys
    
    # --- Create Mock Files for Testing ---
    temp_metrics_csv = "temp_metrics.csv"
    temp_bug_csv = "temp_bug_report.csv"
    
    metrics_data = {
        'filepath': ['f1.c', 'f2.c', 'f3.c', 'f4.c', 'f5.c'],
        'function_name': ['func_a', 'func_b', 'func_c', 'func_d', 'func_e'],
        'LOC': [10, 20, 30, 40, 50],
        'CC': [1, 2, 3, 4, 5]
    }
    pd.DataFrame(metrics_data).to_csv(temp_metrics_csv, index=False)
    pd.DataFrame({'Function_Name_Col': ['func_a', 'func_c'], 'Bug_Status': [1, 1]}).to_csv(temp_bug_csv, index=False)
    
    app = QApplication(sys.argv)
    
    tabs_widget = QTabWidget()
    tab = UseExistingMetricsSubTab()
    tabs_widget.addTab(tab, "Use Existing Metrics")
    
    # Set default values for quick testing
    tab.existing_metrics_input.setText(os.path.abspath(temp_metrics_csv))
    tab.bug_report_input.setText(os.path.abspath(temp_bug_csv))
    tab.output_input.setText(os.path.abspath("final_labeled_output_existing.csv"))
    tab.bug_function_name_input.setText("Function_Name_Col")
    
    tabs_widget.setWindowTitle("Use Existing Metrics Tab (Threaded)")
    tabs_widget.resize(900, 700)
    tabs_widget.show()
    sys.exit(app.exec_())