from PyQt5.QtWidgets import QWidget, QTabWidget, QVBoxLayout
from ui.tabs.subtabs.WPDP.test_scenario_wv import TestScenarioWV
from ui.tabs.subtabs.WPDP.prediction_scenario_wv import PredictionScenarioWV
# from core.models.wpdp_manager import WPDPManager # Placeholder for actual logic

class WPDPWrapperTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("WPDPWrapperTab")

        layout = QVBoxLayout(self)
        sub_tabs = QTabWidget()
        layout.addWidget(sub_tabs)

        # Uses the WV-specific pages
        sub_tabs.addTab(TestScenarioWV(), "Test Scenario")
        sub_tabs.addTab(PredictionScenarioWV(), "Prediction Scenario")