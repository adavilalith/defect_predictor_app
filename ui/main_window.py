import sys
from pathlib import Path
import os
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, 
    QMessageBox, QApplication, QScrollArea, QAction, QLabel
)
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import QEvent, QSettings, Qt
from PyQt5.QtGui import QFont, QPixmap

# --- Imports (Assuming same structure) ---
from ui.tabs.extract_metrics_tab import ExtractMetricsTab
from ui.tabs.wpdp_tab_wrapper import WPDPWrapperTab
from ui.tabs.cvwp_tab_wrapper import CVWPWrapperTab 
from ui.tabs.default_model_tab import DefaultModelTab
from ui.tabs.xai_tab_wrapper import XAIWrapperTab

class DefectPredictionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defect Prediction UI")
        
        # --- 1. Robust Icon & Path Handling ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Path: /home/lalith/DRDL/defect_predictor_app/icon.png
        self.icon_path = os.path.normpath(os.path.join(current_dir, "..", "icon.png"))
        
        if os.path.exists(self.icon_path):
            self.setWindowIcon(QtGui.QIcon(self.icon_path))
        else:
            print(f"Warning: Icon not found at {self.icon_path}")

        # Window Geometry
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.9), int(screen.height() * 0.9))
        self.setMinimumSize(800, 500)

        # Theme/Font Defaults
        self.default_font_size = 10
        self.current_font_size = self.default_font_size

        # Central widget setup
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        central_widget = QWidget()
        scroll_area.setWidget(central_widget)
        self.setCentralWidget(scroll_area)

        self.main_layout = QVBoxLayout(central_widget)

        # --- 2. Create Header with Logo ---
        self.setup_header()

        self.create_menu_bar()
        self.set_light_mode()

        # Tabs container
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.setup_tabs()

    def setup_header(self):
        """Creates a header row containing a title and a larger application icon."""
        header_layout = QHBoxLayout()
        # Add some margin so the large logo doesn't touch the window borders
        header_layout.setContentsMargins(10, 5, 20, 5) 
        
        # 1. Application Title Label
        title_label = QLabel("Software Defect Predictor")
        # I increased the font size slightly to match the bigger icon
        title_label.setStyleSheet("font-size: 22pt; font-weight: bold; color: #008CBA;")
        
        # 2. Logo Label (Top Right)
        self.logo_label = QLabel()
        if os.path.exists(self.icon_path):
            pixmap = QPixmap(self.icon_path)
            
            # --- CHANGE SIZE HERE ---
            # Increased to 80x80. Change these numbers to 100 or 120 if you want it even bigger.
            icon_size = 80 
            
            self.logo_label.setPixmap(pixmap.scaled(
                icon_size, icon_size, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
        
        header_layout.addWidget(title_label)
        header_layout.addStretch() # This pushes the logo to the far right
        header_layout.addWidget(self.logo_label)
        
        # Add the header layout to the main layout before the tabs
        self.main_layout.addLayout(header_layout)

    # --- TAB SETUP (UNCHANGED) ---
    def setup_tabs(self):
        self.tabs.addTab(ExtractMetricsTab(), "Extract Metrics")
        self.tabs.addTab(WPDPWrapperTab(), "WPDP")
        self.tabs.addTab(CVWPWrapperTab(), "CVCPDP")
        self.tabs.addTab(DefaultModelTab(), "Default Model")
        self.tabs.addTab(XAIWrapperTab(), "XAi")

    # --- Menu Bar and Theme Logic ---
    def create_menu_bar(self):
        menu_bar = self.menuBar()
        settings_menu = menu_bar.addMenu("&Settings")
        libclang_action = QAction("Set libclang.so Path...", self)
        libclang_action.triggered.connect(self.set_libclang_path)
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
            QMessageBox.information(self, "Success", f"libclang.so path set to:\n{path}")

    def load_initial_settings(self):
        settings = QSettings("YourOrg", "DefectPrediction")
        self.dark_mode_action.setChecked(False)
        self.set_light_mode()

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
        except Exception as e:
            print(f"Theme Load Error: {e}")

    def toggle_dark_mode(self, checked):
        if checked: self.set_dark_mode()
        else: self.set_light_mode()

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            self.current_font_size = self.default_font_size + 2 if self.isMaximized() else self.default_font_size
            self.update_font_size()
        super().changeEvent(event)

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
        self.tabs.addTab(CVWPWrapperTab(), "CVCPDP")
        
        # 4. CVCP Tab (Cross Version Cross Project)
        # self.tabs.addTab(CVCPWrapperTab(), "CVCP")
        
        # 5. Default Model Tab (Single page)
        self.tabs.addTab(DefaultModelTab(), "Default Model")
        
        # 6. XAI Tab (Wrapper for its 2 sub-tabs)

import sys
from pathlib import Path
import os
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, 
    QMessageBox, QApplication, QScrollArea, QAction, QLabel
)
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import QEvent, QSettings, Qt
from PyQt5.QtGui import QFont, QPixmap

# --- Imports (Assuming same structure) ---
from ui.tabs.extract_metrics_tab import ExtractMetricsTab
from ui.tabs.wpdp_tab_wrapper import WPDPWrapperTab
from ui.tabs.cvwp_tab_wrapper import CVWPWrapperTab 
from ui.tabs.default_model_tab import DefaultModelTab
from ui.tabs.xai_tab_wrapper import XAIWrapperTab

class DefectPredictionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defect Prediction UI")
        
        # --- 1. Robust Icon & Path Handling ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Path: /home/lalith/DRDL/defect_predictor_app/icon.png
        self.icon_path = os.path.normpath(os.path.join(current_dir, "..", "icon.png"))
        
        if os.path.exists(self.icon_path):
            self.setWindowIcon(QtGui.QIcon(self.icon_path))
        else:
            print(f"Warning: Icon not found at {self.icon_path}")

        # Window Geometry
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.9), int(screen.height() * 0.9))
        self.setMinimumSize(800, 500)

        # Theme/Font Defaults
        self.default_font_size = 10
        self.current_font_size = self.default_font_size

        # Central widget setup
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        central_widget = QWidget()
        scroll_area.setWidget(central_widget)
        self.setCentralWidget(scroll_area)

        self.main_layout = QVBoxLayout(central_widget)

        # --- 2. Create Header with Logo ---
        self.setup_header()

        self.create_menu_bar()
        self.set_light_mode()

        # Tabs container
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.setup_tabs()

    def setup_header(self):
        """Creates a header row containing a title and a larger application icon."""
        header_layout = QHBoxLayout()
        # Add some margin so the large logo doesn't touch the window borders
        header_layout.setContentsMargins(10, 5, 20, 5) 
        
        # 1. Application Title Label
        title_label = QLabel("Software Defect Predictor")
        # I increased the font size slightly to match the bigger icon
        title_label.setStyleSheet("font-size: 22pt; font-weight: bold; color: #008CBA;")
        
        # 2. Logo Label (Top Right)
        self.logo_label = QLabel()
        if os.path.exists(self.icon_path):
            pixmap = QPixmap(self.icon_path)
            
            # --- CHANGE SIZE HERE ---
            # Increased to 80x80. Change these numbers to 100 or 120 if you want it even bigger.
            icon_size = 80 
            
            self.logo_label.setPixmap(pixmap.scaled(
                icon_size, icon_size, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
        
        header_layout.addWidget(title_label)
        header_layout.addStretch() # This pushes the logo to the far right
        header_layout.addWidget(self.logo_label)
        
        # Add the header layout to the main layout before the tabs
        self.main_layout.addLayout(header_layout)

    # --- TAB SETUP (UNCHANGED) ---
    def setup_tabs(self):
        self.tabs.addTab(ExtractMetricsTab(), "Extract Metrics")
        self.tabs.addTab(WPDPWrapperTab(), "WPDP")
        self.tabs.addTab(CVWPWrapperTab(), "CVCPDP")
        self.tabs.addTab(DefaultModelTab(), "Default Model")
        self.tabs.addTab(XAIWrapperTab(), "XAi")

    # --- Menu Bar and Theme Logic ---
    def create_menu_bar(self):
        menu_bar = self.menuBar()
        settings_menu = menu_bar.addMenu("&Settings")
        libclang_action = QAction("Set libclang.so Path...", self)
        libclang_action.triggered.connect(self.set_libclang_path)
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
            QMessageBox.information(self, "Success", f"libclang.so path set to:\n{path}")

    def load_initial_settings(self):
        settings = QSettings("YourOrg", "DefectPrediction")
        self.dark_mode_action.setChecked(False)
        self.set_light_mode()

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
        except Exception as e:
            print(f"Theme Load Error: {e}")

    def toggle_dark_mode(self, checked):
        if checked: self.set_dark_mode()
        else: self.set_light_mode()

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            self.current_font_size = self.default_font_size + 2 if self.isMaximized() else self.default_font_size
            self.update_font_size()
        super().changeEvent(event)

    def update_font_size(self):
        font = QFont()
        font.setPointSize(self.current_font_size)
        for widget in self.findChildren(QWidget):
            widget.setFont(font)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Ensure the taskbar icon works on Linux/Windows
    # We set the icon on the application level
    current_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.normpath(os.path.join(current_dir, "..", "icon.png"))
    app.setWindowIcon(QtGui.QIcon(icon_path))
    
    ui = DefectPredictionUI()
    ui.show()
    sys.exit(app.exec_())