"""
graphbert_tab.py
================
GraphCodeBERT feature-extraction sub-tab for the Extract Metrics container.

ROOT CAUSE OF THE FREEZE (and fixes applied)
--------------------------------------------
The original code ran every heavy operation — model loading AND the entire
embedding-generation loop — directly on the Qt main thread.  Qt's event loop
cannot process repaints, button clicks, or any other GUI events while Python
code is executing on that same thread, so the window goes white / unresponsive.
Eventually the OS kills the process when RAM is exhausted loading the model.

Fixes in this rewrite
---------------------
1.  **QThread + Worker objects** – ModelLoaderWorker and EmbeddingWorker run
    in background threads.  They communicate back to the UI exclusively through
    Qt signals (never touching widgets directly), which are automatically
    marshalled to the main thread.

2.  **No QMessageBox / widget calls inside workers** – all user-facing feedback
    is emitted as signals and handled in on-main-thread slots.

3.  **Indeterminate → determinate progress bar** – the bar switches from
    spinning (indeterminate) while the model loads to a real percentage counter
    during embedding generation, so the user always sees progress.

4.  **_load_csv_to_table row-index bug fixed** – the original used the pandas
    index from iterrows() as the QTableWidget row index, which breaks on any
    DataFrame that is not zero-indexed (e.g. after head() / reset).  Now uses
    enumerate().

5.  **Button guard** – Generate / Load buttons are disabled while a worker is
    running to prevent double-submission.
"""

import os
import shutil
import tempfile

import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel
import warnings

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog,
    QComboBox, QFrame, QProgressBar,
    QTableWidget, QTableWidgetItem,
    QSplitter, QMessageBox, QApplication,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

import core.bm0202

warnings.filterwarnings("ignore", message="Some weights of RobertaModel")


# ─────────────────────────────────────────────────────────────────────────────
#  Background worker: load the GraphCodeBERT model
# ─────────────────────────────────────────────────────────────────────────────

class ModelLoaderWorker(QObject):
    """Loads GraphCodeBERT in a background thread."""

    # Emitted when loading succeeds; carries (tokenizer, model, device)
    finished = pyqtSignal(object, object, object)
    # Emitted on failure; carries an error message string
    error = pyqtSignal(str)

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModel.from_pretrained(self.model_path)
            model.to(device)
            model.eval()
            self.finished.emit(tokenizer, model, device)
        except Exception as exc:
            self.error.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
#  Background worker: generate embeddings from a source folder
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingWorker(QObject):
    """
    Parses a source folder, extracts function-level GraphCodeBERT embeddings,
    and saves the result to a temporary CSV file.

    Signals
    -------
    progress(int, int)   – (current, total) for progress-bar updates
    finished(str)        – path to the generated CSV on success
    error(str)           – human-readable error message on failure
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        input_path: str,
        output_name: str,
        libclang_path: str,
        tokenizer,
        model,
        device,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_name = output_name
        self.libclang_path = libclang_path
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    # ------------------------------------------------------------------
    def _get_embedding(self, code_str: str):
        """Return a 768-d numpy embedding for one code string."""
        inputs = self.tokenizer(
            code_str, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    # ------------------------------------------------------------------
    def run(self):
        try:
            # --- Extract functions via libclang ----------------------
            extractor = core.bm0202.MetricsExtractor(self.libclang_path)
            extractor.initialize_libclang()

            if not os.path.isdir(self.input_path):
                self.error.emit(
                    "The input must be a directory containing source code "
                    "for AST parsing."
                )
                return

            functions_list = extractor.extract_functions_from_folder(self.input_path)

            if not functions_list:
                self.error.emit(
                    "No functions found in the selected folder. "
                    "Ensure the folder contains .c / .cpp files."
                )
                return

            total = len(functions_list)
            embeddings_list = []
            metadata_list = []

            for i, func_data in enumerate(functions_list):
                code_str = func_data.get("fCode", "")
                if not code_str or not code_str.strip():
                    self.progress.emit(i + 1, total)
                    continue

                try:
                    emb = self._get_embedding(code_str)
                    embeddings_list.append(emb)
                    metadata_list.append(
                        {"function_name": func_data.get("Function", "")}
                    )
                except Exception as inner:
                    print(
                        f"[EmbeddingWorker] Skipping function "
                        f"'{func_data.get('fName', 'unknown')}': {inner}"
                    )

                # Report progress after every function
                self.progress.emit(i + 1, total)

            if not embeddings_list:
                self.error.emit(
                    "No embeddings were generated. "
                    "Check that the functions contain valid code."
                )
                return

            # --- Build DataFrame and write to a temp file ------------
            emb_cols = [f"emb_{j}" for j in range(768)]
            emb_df = pd.DataFrame(embeddings_list, columns=emb_cols)
            meta_df = pd.DataFrame(metadata_list)
            final_df = pd.concat([meta_df, emb_df], axis=1)

            temp_output_path = os.path.join(tempfile.gettempdir(), self.output_name)
            final_df.to_csv(temp_output_path, index=False)

            self.finished.emit(temp_output_path)

        except Exception as exc:
            self.error.emit(f"Embedding generation failed:\n{exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main sub-tab widget
# ─────────────────────────────────────────────────────────────────────────────

class GraphBERTSubTab(QWidget):
    """
    GraphCodeBERT feature-extraction sub-tab.

    Heavy work (model load, embedding generation) runs in QThread workers.
    The UI remains fully responsive at all times.
    """

    # Hard-coded paths – adjust as needed
    _GRAPHCODEBERT_MODEL_PATH = (
        "/home/omprakash/Desktop/drdl/bug-prediction/"
        "extract_metrics/embeddings_graphbert/graphcodebert-base"
    )
    _PREGENERATED_BASE = (
        "/home/omprakash/Desktop/drdl/bug-prediction/"
        "extract_metrics/embeddings_graphbert/graphbert/graph_embeddings/"
    )
    _LIBCLANG_PATH = "/opt/rh/llvm-toolset-9.0/root/usr/lib64/libclang.so.9"

    def __init__(self, parent=None):
        super().__init__(parent)

        # Loaded model state (set once by ModelLoaderWorker)
        self._tokenizer = None
        self._model = None
        self._device = None

        # Latest generated / loaded CSV path
        self.latest_csv_path = None

        # Active worker thread references (kept to prevent GC)
        self._loader_thread = None
        self._loader_worker = None
        self._embed_thread = None
        self._embed_worker = None

        self._init_ui()

    # ──────────────────────────────────────────────────────────────────
    #  UI construction
    # ──────────────────────────────────────────────────────────────────

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

        # ---- Section 1: Load pre-generated embeddings ----------------
        s1_title = QLabel("View Pre-generated Embeddings")
        s1_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        ctrl.addWidget(s1_title)

        s1_form = QFormLayout()
        s1_form.setSpacing(12)

        dropdown_row = QHBoxLayout()
        self.pregenerated_dropdown = QComboBox()
        self.pregenerated_dropdown.addItems(
            ["p1v1", "p1v2", "p1v3", "p2v1", "p2v2", "p3v1", "p3v2", "p3v3"]
        )
        dropdown_row.addWidget(self.pregenerated_dropdown)

        self.load_btn = QPushButton("Load Embeddings")
        self.load_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "padding: 8px; font-weight: bold;"
        )
        self.load_btn.clicked.connect(self._load_pregenerated_embeddings)

        s1_form.addRow(QLabel("Select Pre-generated Embeddings:"), dropdown_row)
        s1_form.addRow(self.load_btn)
        ctrl.addLayout(s1_form)

        # ---- Separator -----------------------------------------------
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        ctrl.addWidget(sep)

        # ---- Section 2: Generate new embeddings ----------------------
        s2_title = QLabel("Generate New Embeddings (Function Level)")
        s2_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        ctrl.addWidget(s2_title)

        s2_form = QFormLayout()
        s2_form.setSpacing(12)

        browse_row = QHBoxLayout()
        self.input_lineedit = QLineEdit()
        browse_btn = QPushButton("Browse Folder")
        browse_btn.clicked.connect(
            lambda: self._browse_folder(self.input_lineedit)
        )
        browse_row.addWidget(self.input_lineedit)
        browse_row.addWidget(browse_btn)

        self.output_lineedit = QLineEdit()
        self.output_lineedit.setPlaceholderText("e.g. my_embeddings.csv")

        self.generate_btn = QPushButton("Generate Embeddings")
        self.generate_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "padding: 8px; font-weight: bold;"
        )
        self.generate_btn.clicked.connect(self._on_generate_clicked)

        s2_form.addRow(QLabel("Select Source Code Folder:"), browse_row)
        s2_form.addRow(QLabel("Output CSV Name:"), self.output_lineedit)
        s2_form.addRow(self.generate_btn)
        ctrl.addLayout(s2_form)

        ctrl.addStretch()

        # ── Status label ───────────────────────────────────────────────
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #555; font-style: italic;")
        ctrl.addWidget(self.status_label)

        # ── Progress bar ───────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        ctrl.addWidget(self.progress_bar)

        splitter.addWidget(control_widget)

        # ── Preview area ───────────────────────────────────────────────
        self.preview_box = self._build_preview_widget()
        splitter.addWidget(self.preview_box)

        splitter.setSizes([500, 200])
        main_layout.addWidget(splitter)

    def _build_preview_widget(self):
        preview_box = QWidget()
        layout = QVBoxLayout(preview_box)
        layout.setContentsMargins(12, 4, 12, 12)

        self.preview_label = QLabel("Preview of Extracted Embeddings:")
        layout.addWidget(self.preview_label)

        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(5)
        self.preview_table.setHorizontalHeaderLabels(
            ["Col1", "Col2", "Col3", "Col4", "Col5"]
        )
        layout.addWidget(self.preview_table)

        self.download_button = QPushButton("Download CSV")
        self.download_button.clicked.connect(self._download_csv)
        layout.addWidget(self.download_button)

        preview_box.setVisible(False)
        return preview_box

    # ──────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────

    def _browse_folder(self, widget: QLineEdit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            widget.setText(folder)

    def _set_controls_enabled(self, enabled: bool):
        """Disable/enable interactive controls while a worker is running."""
        self.load_btn.setEnabled(enabled)
        self.generate_btn.setEnabled(enabled)
        self.pregenerated_dropdown.setEnabled(enabled)

    def _show_status(self, text: str):
        self.status_label.setText(text)

    # ──────────────────────────────────────────────────────────────────
    #  Section 1 – Load pre-generated embeddings
    # ──────────────────────────────────────────────────────────────────

    def _load_pregenerated_embeddings(self):
        selected = self.pregenerated_dropdown.currentText()
        path = os.path.join(
            self._PREGENERATED_BASE, f"{selected}_embeddings.csv"
        )

        if not os.path.exists(path):
            QMessageBox.warning(
                self, "Warning", f"Embeddings file not found:\n{path}"
            )
            return

        try:
            self.latest_csv_path = path
            self._load_csv_to_table(path)
            QMessageBox.information(
                self, "Success",
                "Embeddings loaded. Click 'Download CSV' to save a copy.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    # ──────────────────────────────────────────────────────────────────
    #  Section 2 – Generate new embeddings  (threaded)
    # ──────────────────────────────────────────────────────────────────

    def _on_generate_clicked(self):
        input_path = self.input_lineedit.text().strip()
        output_name = self.output_lineedit.text().strip() or "output_embeddings.csv"
        if not output_name.endswith(".csv"):
            output_name += ".csv"

        if not input_path:
            QMessageBox.warning(self, "Warning", "Please select a source code folder.")
            return
        if not os.path.exists(input_path):
            QMessageBox.warning(self, "Warning", "The selected path does not exist.")
            return

        # If model is already loaded, go straight to embedding
        if self._model is not None:
            self._start_embedding_worker(input_path, output_name)
            return

        # Otherwise load the model first, then chain into embedding
        self._load_model_then_embed(input_path, output_name)

    # --- Model loading (threaded) -------------------------------------

    def _load_model_then_embed(self, input_path: str, output_name: str):
        """Start ModelLoaderWorker; on success, kick off EmbeddingWorker."""
        self._set_controls_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)          # indeterminate spinner
        self._show_status("Loading GraphCodeBERT model… (this may take a minute)")

        self._loader_worker = ModelLoaderWorker(self._GRAPHCODEBERT_MODEL_PATH)
        self._loader_thread = QThread()
        self._loader_worker.moveToThread(self._loader_thread)

        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_worker.finished.connect(
            lambda tok, mdl, dev: self._on_model_loaded(tok, mdl, dev, input_path, output_name)
        )
        self._loader_worker.error.connect(self._on_model_load_error)

        # Clean up thread when done
        self._loader_worker.finished.connect(self._loader_thread.quit)
        self._loader_worker.error.connect(self._loader_thread.quit)
        self._loader_thread.finished.connect(self._loader_worker.deleteLater)
        self._loader_thread.finished.connect(self._loader_thread.deleteLater)

        self._loader_thread.start()

    def _on_model_loaded(self, tokenizer, model, device, input_path, output_name):
        """Called on the main thread when the model finishes loading."""
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._show_status("Model loaded. Generating embeddings…")
        self._start_embedding_worker(input_path, output_name)

    def _on_model_load_error(self, message: str):
        self.progress_bar.setVisible(False)
        self._set_controls_enabled(True)
        self._show_status("")
        QMessageBox.critical(
            self, "Model Load Error",
            f"Failed to load GraphCodeBERT model:\n{message}"
        )

    # --- Embedding generation (threaded) ------------------------------

    def _start_embedding_worker(self, input_path: str, output_name: str):
        self._set_controls_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)        # will switch to real % later
        self.progress_bar.setValue(0)
        self._show_status("Generating embeddings…")

        self._embed_worker = EmbeddingWorker(
            input_path=input_path,
            output_name=output_name,
            libclang_path=self._LIBCLANG_PATH,
            tokenizer=self._tokenizer,
            model=self._model,
            device=self._device,
        )
        self._embed_thread = QThread()
        self._embed_worker.moveToThread(self._embed_thread)

        self._embed_thread.started.connect(self._embed_worker.run)
        self._embed_worker.progress.connect(self._on_embed_progress)
        self._embed_worker.finished.connect(self._on_embed_finished)
        self._embed_worker.error.connect(self._on_embed_error)

        self._embed_worker.finished.connect(self._embed_thread.quit)
        self._embed_worker.error.connect(self._embed_thread.quit)
        self._embed_thread.finished.connect(self._embed_worker.deleteLater)
        self._embed_thread.finished.connect(self._embed_thread.deleteLater)

        self._embed_thread.start()

    def _on_embed_progress(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
            self._show_status(f"Generating embeddings… {current}/{total} functions")

    def _on_embed_finished(self, csv_path: str):
        self.progress_bar.setVisible(False)
        self._set_controls_enabled(True)
        self._show_status("Done.")

        self.latest_csv_path = csv_path
        self._load_csv_to_table(csv_path)

        try:
            df = pd.read_csv(csv_path)
            num_funcs = len(df)
        except Exception:
            num_funcs = "?"

        QMessageBox.information(
            self, "Success",
            f"Generated embeddings for {num_funcs} function(s).\n"
            "Click 'Download CSV' to save the file."
        )

    def _on_embed_error(self, message: str):
        self.progress_bar.setVisible(False)
        self._set_controls_enabled(True)
        self._show_status("")
        QMessageBox.critical(self, "Embedding Error", message)

    # ──────────────────────────────────────────────────────────────────
    #  Preview table
    # ──────────────────────────────────────────────────────────────────

    def _load_csv_to_table(self, csv_path: str, rows_to_show: int = 100):
        """
        Load (a preview of) a CSV into the QTableWidget.

        BUG FIX: The original code used ``row_idx`` from ``df.iterrows()``,
        which equals the *DataFrame index* (not necessarily 0, 1, 2, …).
        After a ``head()`` on a frame that was previously sliced the index
        can skip values, causing ``setItem`` to silently mis-place rows.
        We now use ``enumerate`` for a guaranteed 0-based row counter.
        """
        try:
            df = pd.read_csv(csv_path)

            display_df = df.head(rows_to_show)
            total_rows = len(df)
            shown_rows = len(display_df)

            if total_rows > rows_to_show:
                self.preview_label.setText(
                    f"Preview (first {shown_rows} of {total_rows} rows):"
                )
            else:
                self.preview_label.setText(f"Preview ({total_rows} rows):")

            self.preview_table.setRowCount(shown_rows)
            self.preview_table.setColumnCount(len(display_df.columns))
            self.preview_table.setHorizontalHeaderLabels(list(display_df.columns))

            # Use enumerate() – NOT the pandas index from iterrows()
            for row_num, (_, row_data) in enumerate(display_df.iterrows()):
                for col_idx, value in enumerate(row_data):
                    self.preview_table.setItem(
                        row_num, col_idx, QTableWidgetItem(str(value))
                    )

            self.preview_table.resizeColumnsToContents()
            self.preview_box.setVisible(True)

        except Exception as exc:
            QMessageBox.warning(self, "CSV Load Error", f"Could not load CSV:\n{exc}")

    # ──────────────────────────────────────────────────────────────────
    #  Download
    # ──────────────────────────────────────────────────────────────────

    def _download_csv(self):
        if not self.latest_csv_path or not os.path.exists(self.latest_csv_path):
            QMessageBox.critical(
                self, "Error",
                "No CSV file available. Please load or generate embeddings first.",
            )
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV File",
            os.path.basename(self.latest_csv_path),
            "CSV Files (*.csv)",
        )
        if save_path:
            try:
                shutil.copy(self.latest_csv_path, save_path)
                QMessageBox.information(
                    self, "Success", f"CSV saved to:\n{save_path}"
                )
            except Exception as exc:
                QMessageBox.critical(self, "Save Error", f"Failed to save CSV:\n{exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  Placeholder for a future Code-T5 sub-tab
# ─────────────────────────────────────────────────────────────────────────────

class CodeT5SubTab(QWidget):
    def __init__(self):
        super().__init__()
        QVBoxLayout(self).addWidget(
            QLabel("UI for 'Code T5 Based Feature Extraction' — to be implemented.")
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Container tab  (unchanged structure; libclang init stays on main thread)
# ─────────────────────────────────────────────────────────────────────────────

from PyQt5.QtWidgets import QTabWidget
from core.metrics_extractor import initialize_clang_library
from ui.tabs.subtabs.metric_extraction.metrics_for_prediction_tab import MetricsExtractionSubTab
from ui.tabs.subtabs.metric_extraction.metrics_bug_label_tab import MetricsBugLabelSubTab
from ui.tabs.subtabs.metric_extraction.existing_metrics_bug_tab import UseExistingMetricsSubTab


class ExtractMetricsTab(QWidget):
    """
    Main container tab for all metrics-extraction and feature-engineering sub-tabs.

    libclang is initialized once here, on the main thread, to avoid
    thread-safety issues in the underlying native library.
    """

    _LIBCLANG_PATH = "/opt/rh/llvm-toolset-9.0/root/usr/lib64/libclang.so.9"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            initialize_clang_library(self._LIBCLANG_PATH)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Clang Initialization Error",
                f"Failed to initialize libclang at:\n'{self._LIBCLANG_PATH}'\n\n"
                f"Extraction will not work.\n\nError: {exc}",
            )

        self._init_ui()

    def _init_ui(self):
        vbox = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)

        self.tabs.addTab(MetricsExtractionSubTab(),  "Extract Metrics for Prediction")
        self.tabs.addTab(MetricsBugLabelSubTab(),     "Extract Metrics and Add Bug Label")
        self.tabs.addTab(UseExistingMetricsSubTab(),  "Use Existing Metrics and Add Bug Label")
        self.tabs.addTab(GraphBERTSubTab(),           "GraphBERT Based Feature Extraction")
        # Uncomment when Code-T5 is implemented:
        # self.tabs.addTab(CodeT5SubTab(), "Code T5 Based Feature Extraction")

        vbox.addWidget(self.tabs)
        self.setLayout(vbox)
