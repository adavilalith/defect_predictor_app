import os
import sys
import json
import shutil
import tempfile
import subprocess
import pandas as pd

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog,
    QComboBox, QFrame, QProgressBar,
    QTableWidget, QTableWidgetItem,
    QSplitter, QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

# ── WATCHER REMAINS THE SAME ────────────────────────────────────────────────

class SubprocessWatcher(QObject):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(str, int)
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
                        self.finished.emit(self._params["output_path"], event["rows"])
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

# ── MAIN TAB WITH YOUR REQUESTED UI ──────────────────────────────────────────

class GraphBERTSubTab(QWidget):
    _LIBCLANG_PATH = "/opt/rh/llvm-toolset-9.0/root/usr/lib64/libclang.so.9"
    _GRAPHCODEBERT_MODEL_PATH = "/home/lalith/DRDL/bug-prediction/extract_metrics/embeddings_graphbert/graphcodebert-base"
    _PREGENERATED_BASE = "/home/lalith/DRDL/bug-prediction/extract_metrics/embeddings_graphbert/graphbert/graph_embeddings/"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.latest_csv_path = None
        self._this_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Critical: Define the App Root so the Worker knows where 'core/' is
        # path: .../defect_predictor_app/ui/tabs/subtabs/metric_extraction/
        # app_root: .../defect_predictor_app/ (4 levels up)
        self._app_root = os.path.abspath(os.path.join(self._this_dir, "../../../../"))
        self._worker_script = os.path.join(self._this_dir, "embedding_subprocess_worker.py")
        
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Vertical)

        # ── Control area ───────────────────────────────────────────────
        control_widget = QWidget()
        ctrl = QVBoxLayout(control_widget)
        ctrl.setSpacing(16)
        ctrl.setContentsMargins(12, 12, 12, 12)

        # Section 1: Load pre-generated
        s1_title = QLabel("View Pre-generated Embeddings")
        s1_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        ctrl.addWidget(s1_title)

        s1_form = QFormLayout()
        dropdown_row = QHBoxLayout()
        self.pregenerated_dropdown = QComboBox()
        self.pregenerated_dropdown.addItems(["p1v1", "p1v2", "p1v3", "p2v1", "p2v2", "p3v1", "p3v2", "p3v3"])
        dropdown_row.addWidget(self.pregenerated_dropdown)

        self.load_btn = QPushButton("Load Embeddings")
        self.load_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px; font-weight: bold;")
        self.load_btn.clicked.connect(self._load_pregenerated_embeddings)

        s1_form.addRow(QLabel("Select Pre-generated Embeddings:"), dropdown_row)
        s1_form.addRow(self.load_btn)
        ctrl.addLayout(s1_form)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFrameShadow(QFrame.Sunken); ctrl.addWidget(sep)

        # Section 2: Generate new
        s2_title = QLabel("Generate New Embeddings (Function Level)")
        s2_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        ctrl.addWidget(s2_title)

        s2_form = QFormLayout()
        browse_row = QHBoxLayout()
        self.input_lineedit = QLineEdit()
        browse_btn = QPushButton("Browse Folder")
        browse_btn.clicked.connect(lambda: self._browse_folder(self.input_lineedit))
        browse_row.addWidget(self.input_lineedit)
        browse_row.addWidget(browse_btn)

        self.output_lineedit = QLineEdit()
        self.output_lineedit.setPlaceholderText("e.g. my_embeddings.csv")

        self.generate_btn = QPushButton("Generate Embeddings")
        self.generate_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px; font-weight: bold;")
        self.generate_btn.clicked.connect(self._on_generate_clicked)

        s2_form.addRow(QLabel("Select Source Code Folder:"), browse_row)
        s2_form.addRow(QLabel("Output CSV Name:"), self.output_lineedit)
        s2_form.addRow(self.generate_btn)
        ctrl.addLayout(s2_form)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #555; font-style: italic;")
        ctrl.addWidget(self.status_label)

        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        ctrl.addWidget(self.progress_bar)
        splitter.addWidget(control_widget)

        # Preview area
        self.preview_box = self._build_preview_widget()
        splitter.addWidget(self.preview_box)
        main_layout.addWidget(splitter)

    def _build_preview_widget(self):
        w = QWidget(); l = QVBoxLayout(w)
        self.preview_label = QLabel("Preview of Extracted Embeddings:")
        self.preview_table = QTableWidget()
        self.download_button = QPushButton("Download CSV")
        self.download_button.clicked.connect(self._download_csv)
        l.addWidget(self.preview_label); l.addWidget(self.preview_table); l.addWidget(self.download_button)
        w.setVisible(False)
        return w

    # ── LOGIC ──

    def _browse_folder(self, edit):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path: edit.setText(path)

    def _on_generate_clicked(self):
        in_path = self.input_lineedit.text().strip()
        if not in_path: return

        out_name = self.output_lineedit.text().strip() or "output_embeddings.csv"
        out_path = os.path.join(tempfile.gettempdir(), out_name)

        params = {
            "app_root": self._app_root,        # Passed to worker for sys.path
            "input_path": in_path,
            "output_path": out_path,
            "libclang_path": self._LIBCLANG_PATH,
            "model_path": self._GRAPHCODEBERT_MODEL_PATH
        }

        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Worker starting...")

        self.thread = QThread()
        self.worker = SubprocessWatcher(params, self._worker_script, sys.executable)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._update_progress)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.thread.start()

    def _update_progress(self, curr, total):
        self.progress_bar.setRange(0, total); self.progress_bar.setValue(curr)
        self.status_label.setText(f"Processing function {curr}/{total}")

    def _on_done(self, path, rows):
        self.generate_btn.setEnabled(True); self.progress_bar.setVisible(False)
        self.latest_csv_path = path
        self._load_csv_to_table(path)
        QMessageBox.information(self, "Success", f"Processed {rows} functions.")

    def _on_error(self, msg):
        self.generate_btn.setEnabled(True); self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", msg)

    def _load_csv_to_table(self, path):
        df = pd.read_csv(path).head(100)
        self.preview_table.setRowCount(len(df)); self.preview_table.setColumnCount(len(df.columns))
        self.preview_table.setHorizontalHeaderLabels(list(df.columns))
        for i, (idx, row) in enumerate(df.iterrows()):
            for j, val in enumerate(row):
                self.preview_table.setItem(i, j, QTableWidgetItem(str(val)))
        self.preview_box.setVisible(True)

    def _load_pregenerated_embeddings(self):
        sel = self.pregenerated_dropdown.currentText()
        path = os.path.join(self._PREGENERATED_BASE, f"{sel}_embeddings.csv")
        if os.path.exists(path):
            self.latest_csv_path = path
            self._load_csv_to_table(path)
        else:
            QMessageBox.warning(self, "Error", "File not found.")

    def _download_csv(self):
        if not self.latest_csv_path: return
        dest, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV (*.csv)")
        if dest: shutil.copy(self.latest_csv_path, dest)