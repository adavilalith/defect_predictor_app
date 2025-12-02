from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt

class PredictionScenarioCV(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #fff3e0; border: 1px dashed #ffb74d;")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("CVWP/CVCP: Prediction Scenario (CV - SHARED LAYOUT)", alignment=Qt.AlignCenter))
        layout.addWidget(QLabel("This layout is shared for Cross-Version and Cross-Project prediction.", alignment=Qt.AlignCenter))