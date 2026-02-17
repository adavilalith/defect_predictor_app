"""
Template for adding reset functionality to subtabs.

This template shows how to add reset functionality to any subtab in the application.
Follow these steps:

1. Import the ResetMixin
2. Inherit from ResetMixin in your class
3. Call enable_reset_button() when output is generated
4. Implement reset_to_defaults() method

Example implementation:
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QPushButton
from ui.components.reset_mixin import ResetMixin


class ExampleSubTab(QWidget, ResetMixin):
    """Example subtab with reset functionality."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Your UI elements
        self.input_field = QLineEdit()
        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self.process_data)
        
        layout.addWidget(self.input_field)
        layout.addWidget(self.process_btn)
        
        # Add reset button (optional - can be added at wrapper level)
        # self.setup_reset_button(layout)
    
    def process_data(self):
        """Process data and enable reset button when done."""
        # Your processing logic here
        
        # Enable reset button after successful processing
        self.enable_reset_button()
    
    def reset_to_defaults(self):
        """Reset all parameters and input files to default values."""
        # Clear input fields
        self.input_field.clear()
        
        # Reset any other UI elements to their default state
        # self.checkbox.setChecked(False)
        # self.combo_box.setCurrentIndex(0)
        # self.slider.setValue(default_value)
        
        # Clear any results/output displays
        # self.results_table.setRowCount(0)
        # self.results_area.setVisible(False)
        
        # Reset progress bars
        # self.progress_bar.setValue(0)
        
        # Reset UI state if you have a set_ui_state method
        # self.set_ui_state(True)


"""
For wrapper tabs (tabs that contain subtabs), the pattern is:

class ExampleWrapperTab(QWidget, ResetMixin):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        self.sub_tabs = QTabWidget()
        self.subtab1 = SubTab1()
        self.subtab2 = SubTab2()
        
        self.sub_tabs.addTab(self.subtab1, "Tab 1")
        self.sub_tabs.addTab(self.subtab2, "Tab 2")
        
        layout.addWidget(self.sub_tabs)
        
        # Add reset button
        self.setup_reset_button(layout)
    
    def reset_to_defaults(self):
        # Reset each subtab
        if hasattr(self.subtab1, 'reset_to_defaults'):
            self.subtab1.reset_to_defaults()
        if hasattr(self.subtab2, 'reset_to_defaults'):
            self.subtab2.reset_to_defaults()
        
        # Switch to first tab
        self.sub_tabs.setCurrentIndex(0)
"""