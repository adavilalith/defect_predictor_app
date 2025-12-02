import sys
from pathlib import Path

# Add project root to Python path (Ensures imports from 'ui.tabs' work)
# This line is crucial for finding modules in ui/tabs/
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QFileDialog, 
    QMessageBox, QApplication, QScrollArea, QAction
)
from PyQt5.QtCore import QEvent, QSettings
from PyQt5.QtGui import QFont

# --- REFACTORED IMPORTS: Clean and fully namespaced ---
from ui.tabs.extract_metrics_tab import ExtractMetricsTab
from ui.tabs.wpdp_tab_wrapper import WPDPWrapperTab
from ui.tabs.cvwp_tab_wrapper import CVWPWrapperTab # Renamed from CVDP
from ui.tabs.cvcp_tab_wrapper import CVCPWrapperTab # Renamed from CPDP
from ui.tabs.default_model_tab import DefaultModelTab
from ui.tabs.xai_tab_wrapper import XAIWrapperTab

class DefectPredictionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defect Prediction UI")

        # Dynamic resizing and safety minimums
        screen = QApplication.primaryScreen().availableGeometry()
        width = int(screen.width() * 0.9)
        height = int(screen.height() * 0.9)
        self.resize(width, height)
        self.setMinimumSize(800, 500)

        self.default_font_size = 10
        self.current_font_size = self.default_font_size

        # Central widget setup (with scroll area)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        central_widget = QWidget()
        scroll_area.setWidget(central_widget)
        self.setCentralWidget(scroll_area)

        self.main_layout = QVBoxLayout(central_widget)

        # Menu and theme setup
        self.create_menu_bar()
        self.set_light_mode()

        # Tabs container
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.setup_tabs()

    # --- Menu Bar and Settings Methods (UNCHANGED) ---
    def create_menu_bar(self):
        menu_bar = self.menuBar()
        settings_menu = menu_bar.addMenu("&Settings")
        libclang_action = QAction("Set libclang.so Path...", self)
        libclang_action.triggered.connect(lambda: self.set_libclang_path())
        settings_menu.addAction(libclang_action)
        view_menu = menu_bar.addMenu("&View")
        self.dark_mode_action = QAction("Dark Mode", self)
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.toggled.connect(self.toggle_dark_mode)
        view_menu.addAction(self.dark_mode_action)
        self.load_initial_settings()

    def set_libclang_path(self):
        path, ok = QFileDialog.getOpenFileName(self, "Select libclang.so", "", "Shared Library (*.so *.dll *.dylib)")
        if ok and path:
            settings = QSettings("YourOrg", "DefectPrediction")
            settings.setValue("libclang_path", path)
            self.libclang_path = path
            QMessageBox.information(self, "Success", f"libclang.so path set to:\n{path}")

    def load_initial_settings(self):
        settings = QSettings("YourOrg", "DefectPrediction")
        libclang_path = settings.value("libclang_path", "")
        if libclang_path:
            self.libclang_path = libclang_path
        self.dark_mode_action.setChecked(False)
        self.set_light_mode()
        settings.setValue("dark_mode", False)
        settings.sync()

    def set_light_mode(self):
        theme_path = Path(__file__).parent.parent / "styles" / "lightMode.qss"
        self._load_theme(theme_path)

    def set_dark_mode(self):
        theme_path = Path(__file__).parent.parent / "styles" / "darkMode.qss"
        self._load_theme(theme_path)

    def _load_theme(self, theme_path):
        try:
            with open(theme_path, 'r') as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print(f"Warning: Theme file not found at {theme_path}")
            if "darkMode" in str(theme_path):
                self.setStyleSheet("QMainWindow { background: #333; color: white; }")
            else:
                self.setStyleSheet("QMainWindow { background: white; color: black; }")
        except Exception as e:
            print(f"Error loading theme: {str(e)}")
            self.setStyleSheet("")

    def toggle_dark_mode(self, checked):
        if checked:
            self.set_dark_mode()
        else:
            self.set_light_mode()
        settings = QSettings("YourOrg", "DefectPrediction")
        settings.setValue("dark_mode", checked)

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            if self.isMaximized():
                self.current_font_size = self.default_font_size + 2
            else:
                self.current_font_size = self.default_font_size
            self.update_font_size()
        super(DefectPredictionUI, self).changeEvent(event)

    def update_font_size(self):
        font = QFont()
        font.setPointSize(self.current_font_size)
        for widget in self.findChildren(QWidget):
            widget.setFont(font)

    # --- REFACTORED setup_tabs() ---
    def setup_tabs(self):
        """
        Creates and adds all main tabs using their dedicated wrapper classes.
        Each wrapper handles its own nested QTabWidgets internally.
        """
        
        # 1. Extract Metrics Tab (Wrapper for its 5 sub-tabs)
        self.tabs.addTab(ExtractMetricsTab(), "Extract Metrics")
        
        # 2. WPDP Tab (Wrapper for Test/Prediction sub-tabs)
        self.tabs.addTab(WPDPWrapperTab(), "WPDP")
        
        # 3. CVWP Tab (Cross Version Within Project)
        self.tabs.addTab(CVWPWrapperTab(), "CVWP")
        
        # 4. CVCP Tab (Cross Version Cross Project)
        self.tabs.addTab(CVCPWrapperTab(), "CVCP")
        
        # 5. Default Model Tab (Single page)
        self.tabs.addTab(DefaultModelTab(), "Default Model")
        
        # 6. XAI Tab (Wrapper for its 2 sub-tabs)
        self.tabs.addTab(XAIWrapperTab(), "XAi")