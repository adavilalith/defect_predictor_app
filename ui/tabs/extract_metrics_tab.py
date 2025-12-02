from PyQt5.QtWidgets import QWidget, QTabWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

# Placeholder factory for the 5 unique metric sub-tabs
def create_metric_placeholder(name):
    class Placeholder(QWidget):
        def __init__(self, p=None):
            super().__init__(p)
            self.setStyleSheet("background-color: #ffebee; border: 1px solid #e57373;")
            l = QVBoxLayout(self)
            label = QLabel(f"METRIC EXTRACTION: {name}", alignment=Qt.AlignCenter)
            l.addWidget(label)
    return Placeholder()

class ExtractMetricsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ExtractMetricsTab")

        main_layout = QVBoxLayout(self)
        self.sub_tabs = QTabWidget()
        main_layout.addWidget(self.sub_tabs)

        # Add the 5 Sub-Tabs
        self.sub_tabs.addTab(create_metric_placeholder("Extract Metrics for Prediction"), "Metrics for Prediction")
        self.sub_tabs.addTab(create_metric_placeholder("Extract Metrics _Add Bug Label"), "Metrics + Bug Label")
        self.sub_tabs.addTab(create_metric_placeholder("Use Existing Metrics _Add Bug Label"), "Existing Metrics + Bug")
        self.sub_tabs.addTab(create_metric_placeholder("GraphBERT Based Feature Extraction"), "GraphBERT Features")
        self.sub_tabs.addTab(create_metric_placeholder("Code T5 Based Feature Extraction"), "Code T5 Features") 