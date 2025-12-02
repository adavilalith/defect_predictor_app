from PyQt5.QtWidgets import QWidget, QTabWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

# ----------------------------------------------------------------------
# Helper placeholders for the XAI Scenario Sub-pages
# ----------------------------------------------------------------------

# Placeholder for ui/tabs/subtabs/xai_scenarios/within_version_xai.py
class WithinVersionXAI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #fce4ec; border: 1px solid #ec407a;")
        l = QVBoxLayout(self)
        l.addWidget(QLabel("XAI SCENARIO: With Version Within Project", alignment=Qt.AlignCenter))

# Placeholder for ui/tabs/subtabs/xai_scenarios/cross_version_xai.py
class CrossVersionXAI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #e1f5fe; border: 1px solid #29b6f6;")
        l = QVBoxLayout(self)
        l.addWidget(QLabel("XAI SCENARIO: Cross Version Cross Project", alignment=Qt.AlignCenter))

# ----------------------------------------------------------------------
# Main XAI Wrapper Class
# ----------------------------------------------------------------------

class XAIWrapperTab(QWidget):
    """
    Wrapper for the XAI main tab, containing two nested scenario tabs.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("XAIWrapperTab")
        
        layout = QVBoxLayout(self)
        sub_tabs = QTabWidget()
        layout.addWidget(sub_tabs)
        
        # Add the two XAI specific sub-tabs
        sub_tabs.addTab(WithinVersionXAI(), "With Version Within Project")
        sub_tabs.addTab(CrossVersionXAI(), "Cross Version Cross Project")