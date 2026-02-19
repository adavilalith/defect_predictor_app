import os
import sys
import json
import subprocess
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLineEdit, QFileDialog, QProgressBar, 
    QMessageBox, QScrollArea, QTableWidget, QTableWidgetItem,
    QSizePolicy, QLabel, QComboBox, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

# Assuming these exist in your project structure
from ui.components.reset_mixin import ResetMixin 

# --- 1. Subprocess Watcher ---
class SubprocessWatcher(QObject):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(pd.DataFrame)
    error    = pyqtSignal(str)

    def __init__(self, params, worker_script, python_exe):
        super().__init__()
        self._params = params
        self._worker_script = worker_script
        self._python_exe = python_exe

    def run(self):
        try:
            proc = subprocess.Popen(
                [self._python_exe, self._worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            proc.stdin.write(json.dumps(self._params))
            proc.stdin.close()

            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line: continue
                try:
                    event = json.loads(line)
                    if event["type"] == "progress":
                        self.progress.emit(event["current"], event["total"])
                    elif event["type"] == "done":
                        if os.path.exists(self._params["output_path"]):
                            df = pd.read_csv(self._params["output_path"])
                            self.finished.emit(df)
                        else:
                            self.error.emit("Worker finished but output file not found.")
                        return
                    elif event["type"] == "error":
                        self.error.emit(event["message"])
                        return
                except: continue

            proc.wait()
            if proc.returncode != 0:
                self.error.emit(f"Worker Error: {proc.stderr.read()}")
        except Exception as e:
            self.error.emit(str(e))

# --- 2. Main Tab ---
class GraphBERTSubTab(QWidget, ResetMixin):
    _LIBCLANG_PATH = "/opt/rh/llvm-toolset-9.0/root/usr/lib64/libclang.so.9"
    _MODEL_PATH = "/home/lalith/DRDL/bug-prediction/extract_metrics/embeddings_graphbert/graphcodebert-base"
    _PREGEN_BASE = "/home/lalith/DRDL/bug-prediction/extract_metrics/embeddings_graphbert/graphbert/graph_embeddings/"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._this_dir = os.path.dirname(os.path.abspath(__file__))
        self._app_root = os.path.abspath(os.path.join(self._this_dir, "../../../../"))
        self._worker_script = os.path.join(self._this_dir, "embedding_subprocess_worker.py")
        
        self.df_result = None
        self.init_ui()

    def init_ui(self):
        # Main Layout
        self.main_vbox = QVBoxLayout(self)
        self.main_vbox.setSpacing(10)
        self.main_vbox.setContentsMargins(10, 10, 10, 10)

        # --- Section: Pre-generated ---
        pregen_layout = QHBoxLayout()
        self.pregenerated_dropdown = QComboBox()
        self.pregenerated_dropdown.addItems(["p1v1", "p1v2", "p1v3", "p2v1", "p2v2", "p3v1", "p3v2", "p3v3"])
        self.load_pregen_btn = QPushButton("Load Pre-generated")
        
        pregen_layout.addWidget(QLabel("Pre-generated:"))
        pregen_layout.addWidget(self.pregenerated_dropdown)
        pregen_layout.addWidget(self.load_pregen_btn)
        pregen_layout.addStretch()
        self.main_vbox.addLayout(pregen_layout)

        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken)
        self.main_vbox.addWidget(line)

        # --- Section: Generation Inputs ---
        # Folder Selection
        h_folder = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Select Source Code Folder...")
        self.browse_folder_btn = QPushButton("Browse Folder")
        h_folder.addWidget(self.folder_input)
        h_folder.addWidget(self.browse_folder_btn)
        self.main_vbox.addLayout(h_folder)

        # Output Selection
        h_output = QHBoxLayout()
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Select Output CSV File Path...")
        self.browse_output_btn = QPushButton("Browse Output")
        h_output.addWidget(self.output_input)
        h_output.addWidget(self.browse_output_btn)
        self.main_vbox.addLayout(h_output)

        # --- Section: Action Buttons ---
        button_layout = QHBoxLayout()
        self.extract_btn = QPushButton("Generate Embeddings and Save")
        self.extract_btn.setStyleSheet("background-color: #007ACC; color: white;")
        button_layout.addWidget(self.extract_btn)
        
        self.setup_reset_button(button_layout)
        self.main_vbox.addLayout(button_layout)

        # --- Section: Progress & Results ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.main_vbox.addWidget(self.progress_bar)

        self.results_table = QTableWidget()
        self.results_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.results_table)
        self.scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_area.setVisible(False)
        self.main_vbox.addWidget(self.scroll_area)

        # This stretch is the key: it stays at 0 when table is visible, 
        # but pushes everything up when the table is hidden.
        self.main_vbox.addStretch(1)

        # --- Signals ---
        self.load_pregen_btn.clicked.connect(self._load_pregenerated)
        self.browse_folder_btn.clicked.connect(self._select_source_folder)
        self.browse_output_btn.clicked.connect(self._select_output_file)
        self.extract_btn.clicked.connect(self.start_generation)

    def set_ui_state(self, enabled):
        self.folder_input.setEnabled(enabled)
        self.browse_folder_btn.setEnabled(enabled)
        self.output_input.setEnabled(enabled)
        self.browse_output_btn.setEnabled(enabled)
        self.extract_btn.setEnabled(enabled)
        self.pregenerated_dropdown.setEnabled(enabled)
        self.load_pregen_btn.setEnabled(enabled)

        if enabled:
            self.enable_reset_button()
            self.extract_btn.setText("Generate Embeddings and Save")
            self.extract_btn.setStyleSheet("background-color: #007ACC; color: white;")
        else:
            self.disable_reset_button()
            self.extract_btn.setText("Processing...")
            self.extract_btn.setStyleSheet("background-color: #FFC107; color: black;")

    def _select_source_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Source Folder", os.getcwd())
        if path: self.folder_input.setText(path)

    def _select_output_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Metrics CSV", "embeddings_output.csv", "CSV Files (*.csv)")
        if path: self.output_input.setText(path)

    def start_generation(self):
        in_path = self.folder_input.text().strip()
        out_path = self.output_input.text().strip()

        if not in_path or not os.path.isdir(in_path):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid source folder.")
            return
        if not out_path:
            QMessageBox.warning(self, "Invalid Input", "Please select an output path.")
            return

        params = {
            "app_root": self._app_root,
            "input_path": in_path,
            "output_path": out_path,
            "libclang_path": self._LIBCLANG_PATH,
            "model_path": self._MODEL_PATH
        }

        self.set_ui_state(False)
        self.progress_bar.setValue(0)
        self.scroll_area.setVisible(False)

        self.thread = QThread()
        self.worker = SubprocessWatcher(params, self._worker_script, sys.executable)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._update_progress_bar)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.thread.start()

    def _update_progress_bar(self, curr, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(curr)

    def _on_finished(self, df):
        self.thread.quit()
        self.df_result = df
        self.set_ui_state(True)
        self.preview_results(df)
        QMessageBox.information(self, "Success", f"Embeddings saved to:\n{self.output_input.text()}")

    def _on_error(self, msg):
        self.thread.quit()
        self.set_ui_state(True)
        QMessageBox.critical(self, "Error", msg)

    def preview_results(self, df):
        if df is None or df.empty:
            self.scroll_area.setVisible(False)
            return

        df_preview = df.head(50)
        self.results_table.setColumnCount(len(df_preview.columns))
        self.results_table.setHorizontalHeaderLabels(df_preview.columns)
        self.results_table.setRowCount(len(df_preview))
        
        for i, (idx, row) in enumerate(df_preview.iterrows()):
            for j, col in enumerate(df_preview.columns):
                val = row[col]
                text = f"{val:.4f}" if isinstance(val, float) else str(val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(i, j, item)
        
        self.results_table.resizeColumnsToContents()
        self.scroll_area.setVisible(True)

    def _load_pregenerated(self):
        sel = self.pregenerated_dropdown.currentText()
        path = os.path.join(self._PREGEN_BASE, f"{sel}_embeddings.csv")
        if os.path.exists(path):
            self.df_result = pd.read_csv(path)
            self.preview_results(self.df_result)
        else:
            QMessageBox.warning(self, "Error", f"File not found: {path}")

    def reset_to_defaults(self):
        self.folder_input.clear()
        self.output_input.clear()
        self.progress_bar.setValue(0)
        self.df_result = None
        self.results_table.setRowCount(0)
        self.scroll_area.setVisible(False)
        self.set_ui_state(True)