"""
Reset functionality mixin for UI tabs.
Provides a standardized way to add reset buttons to tabs.
"""
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QWidget, QVBoxLayout, QMessageBox
from PyQt5.QtCore import pyqtSignal


class ResetMixin:
    """
    Mixin class that provides reset functionality for UI tabs.
    
    Usage:
    1. Inherit from this mixin in your tab class
    2. Call setup_reset_button() in your init_ui method
    3. Implement reset_to_defaults() method to define reset behavior
    4. Call enable_reset_button() when output is generated
    """
    
    # Signal emitted when reset is performed
    reset_performed = pyqtSignal()
    
    def setup_reset_button(self, button_layout):
        """
        Sets up the reset button and adds it to the button layout (beside start/execute button).
        
        Args:
            button_layout: The layout where the reset button should be added (usually beside the main action button)
        """
        # Create reset button
        self.reset_btn = QPushButton(" Reset")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5722; 
                color: white; 
                padding: 8px 12px; 
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        self.reset_btn.setEnabled(True)  # Enabled by default
        self.reset_btn.setToolTip("Reset all parameters and clear input files")
        self.reset_btn.clicked.connect(self.perform_reset)
        
        # Add reset button to the provided layout
        button_layout.addWidget(self.reset_btn)
        
        # Set stretch factors: main button gets 70%, reset button gets 30%
        if button_layout.count() >= 2:
            button_layout.setStretchFactor(button_layout.itemAt(0).widget(), 7)  # 70%
            button_layout.setStretchFactor(button_layout.itemAt(1).widget(), 3)  # 30%
    
    def enable_reset_button(self):
        """Enable the reset button (call this when processing completes)"""
        if hasattr(self, 'reset_btn'):
            self.reset_btn.setEnabled(True)
    
    def disable_reset_button(self):
        """Disable the reset button (call this when processing starts)"""
        if hasattr(self, 'reset_btn'):
            self.reset_btn.setEnabled(False)
    
    def perform_reset(self):
        """
        Performs the reset operation with user confirmation.
        """
        # Ask for confirmation
        reply = QMessageBox.question(
            self, 
            'Confirm Reset', 
            'Are you sure you want to reset all parameters and clear input files?\n\nThis action cannot be undone.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Call the implemented reset method
                self.reset_to_defaults()
                
                # Disable reset button after reset
                self.disable_reset_button()
                
                # Re-enable reset button after a short delay (since we just reset)
                # This ensures the button is available again after reset
                self.enable_reset_button()
                
                # Emit signal
                if hasattr(self, 'reset_performed'):
                    self.reset_performed.emit()
                
                QMessageBox.information(self, "Reset Complete", "All parameters have been reset to default values.")
                
            except Exception as e:
                QMessageBox.critical(self, "Reset Error", f"An error occurred during reset:\n{str(e)}")
    
    def reset_to_defaults(self):
        """
        Override this method in your tab class to implement specific reset behavior.
        This method should:
        1. Clear all input fields
        2. Reset all parameters to default values
        3. Clear any output displays
        4. Reset UI state
        """
        raise NotImplementedError("Subclasses must implement reset_to_defaults method")