import sys
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QMessageBox, QLabel
)
from core.metrics_extractor import initialize_clang_library # Function for one-time setup
from ui.tabs.subtabs.metric_extraction.metrics_for_prediction_tab import MetricsExtractionSubTab
from ui.tabs.subtabs.metric_extraction.metrics_bug_label_tab import MetricsBugLabelSubTab

# --- Placeholder Classes for Future Sub-Tabs ---
# These classes are temporary wrappers to allow the main tab structure to compile.
# You will replace these with the actual implementation files later.

class ExistingMetricsBugLabelSubTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("UI for 'Use Existing Metrics _Add Bug Label' to be implemented here."))

class GraphBERTSubTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("UI for 'GraphBERT Based Feature Extraction' to be implemented here."))

class CodeT5SubTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("UI for 'Code T5 Based Feature Extraction' to be implemented here."))

# ----------------------------------------------------

class ExtractMetricsTab(QWidget):
    """
    The main container tab for all metrics extraction and feature engineering sub-tabs.
    CRITICAL: It performs the one-time libclang initialization to prevent thread-safety errors.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # --- 1. CRITICAL: Initialize libclang ONCE in the main thread ---
        # This path is based on the system configuration noted in previous steps.
        self.libclang_path = "/opt/rh/llvm-toolset-9.0/root/usr/lib64/libclang.so.9"
        
        try:
            # Call the standalone initialization function
            initialize_clang_library(self.libclang_path)
        except Exception as e:
            # Display a critical error if initialization fails
            QMessageBox.critical(self, "Clang Initialization Error", 
                                 f"Failed to initialize libclang at path '{self.libclang_path}'. "
                                 f"Extraction will not work. Error: {e}")

        self.init_ui()

    def init_ui(self):
        vbox = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North) 
        
        # --- 2. Create and Add Sub-Tabs (Matching User's Hierarchy) ---
        
        # 2.1 Extract Metrics for Prediction (Your existing implemented sub-tab)
        self.extraction_for_prediction_tab = MetricsExtractionSubTab()
        self.tabs.addTab(self.extraction_for_prediction_tab, "Extract Metrics for Prediction")
        
        # 2.2 Extract Metrics _Add Bug Label (Placeholder)
        self.bug_label_tab = MetricsBugLabelSubTab()
        self.tabs.addTab(self.bug_label_tab, "Extract Metrics _Add Bug Label")
        
        # 2.3 Use Existing Metrics _Add Bug Label (Placeholder)
        self.existing_bug_label_tab = ExistingMetricsBugLabelSubTab()
        self.tabs.addTab(self.existing_bug_label_tab, "Use Existing Metrics _Add Bug Label")

        # 2.4 GraphBERT Based Feature Extraction (Placeholder)
        self.graphbert_tab = GraphBERTSubTab()
        self.tabs.addTab(self.graphbert_tab, "GraphBERT Based Feature Extraction")

        # 2.5 Code T5 Based Feature Extraction (Placeholder)
        self.codet5_tab = CodeT5SubTab()
        self.tabs.addTab(self.codet5_tab, "Code T5 Based Feature Extraction")

        vbox.addWidget(self.tabs)
        self.setLayout(vbox)