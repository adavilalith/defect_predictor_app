from PyQt5.QtWidgets import QWidget, QTabWidget, QVBoxLayout
from ui.tabs.subtabs.CVWP.test_scenario_cv import TestScenarioCV
from ui.tabs.subtabs.CVWP.prediction_scenario_cv import PredictionScenarioCV

class CVWPWrapperTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CVWPWrapperTab")
        layout = QVBoxLayout(self)
        self.sub_tabs = QTabWidget()
        layout.addWidget(self.sub_tabs)

        # Uses the CV-shared pages
        self.test_tab = TestScenarioCV()
        self.prediction_tab = PredictionScenarioCV()
        
        self.sub_tabs.addTab(self.test_tab, "Test Scenario")
        self.sub_tabs.addTab(self.prediction_tab, "Prediction Scenario")