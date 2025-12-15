from PyQt5.QtWidgets import QWidget, QTabWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from ui.tabs.subtabs.xai.within_version_xai import WithinVersionXAI
from ui.tabs.subtabs.xai.cross_version_xai import CrossVersionXAI
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