from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt

class DefaultModelTab(QWidget):
    """
    Placeholder for the single-page Default Model Tab.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DefaultModelTab")
        self.setStyleSheet("background-color: #e8f5e9;")
        
        layout = QVBoxLayout(self)
        
        label = QLabel("DEFAULT MODEL TAB\n(Simple QWidget - No Sub-tabs)")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 16pt; color: #43a047;")
        
        layout.addWidget(label)