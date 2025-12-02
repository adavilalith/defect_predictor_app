from PyQt5.QtWidgets import QWidget, QTabWidget, QVBoxLayout
from ui.tabs.subtabs.shared_prediction_cv.test_scenario_cv import TestScenarioCV
from ui.tabs.subtabs.shared_prediction_cv.prediction_scenario_cv import PredictionScenarioCV

class CVCPWrapperTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CVWPWrapperTab")
        layout = QVBoxLayout(self)
        sub_tabs = QTabWidget()
        layout.addWidget(sub_tabs)

        # Uses the CV-shared pages
        sub_tabs.addTab(TestScenarioCV(), "Test Scenario")
        sub_tabs.addTab(PredictionScenarioCV(), "Prediction Scenario")