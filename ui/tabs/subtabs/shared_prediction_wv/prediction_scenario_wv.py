from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt

class PredictionScenarioWV(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #e3f2fd; border: 1px dashed #64b5f6;")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("WPDP: Prediction Scenario (WV - UNIQUE LAYOUT)", alignment=Qt.AlignCenter))
        layout.addWidget(QLabel("This layout is specific to Within-Version defect prediction.", alignment=Qt.AlignCenter))