from PyQt5.QtWidgets import (
    QWidget, QFormLayout, QLabel, QLineEdit, QPushButton,
    QCheckBox, QComboBox, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QTextEdit, QScrollArea, QDialog,
    QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox,
    QProgressBar, QSplitter, QApplication, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent
from PyQt5.QtGui import QFont, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar 
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import os


class ZoomableCanvas(FigureCanvas):
    """Custom FigureCanvas with zoom functionality"""
    def __init__(self, figure, parent=None):
        super().__init__(figure)
        self.setParent(parent)
        self.zoom_factor = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 5.0
        self.base_width = 800
        self.base_height = 600
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        
        # Install event filter to capture wheel events
        self.installEventFilter(self)
        
    def eventFilter(self, obj, event):
        """Filter and handle wheel events for zooming"""
        if event.type() == QEvent.Wheel and obj == self:
            # Get the wheel delta (positive = zoom in, negative = zoom out)
            delta = event.angleDelta().y()
            
            # Calculate zoom factor change
            zoom_change = 1.15 if delta > 0 else 1/1.15
            
            # Update zoom factor with limits
            new_zoom = self.zoom_factor * zoom_change
            new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
            
            if new_zoom != self.zoom_factor:
                self.zoom_factor = new_zoom
                self.apply_zoom()
            
            return True  # Event handled
        
        return super().eventFilter(obj, event)
    
    def apply_zoom(self):
        """Apply the current zoom factor to the canvas size"""
        new_width = int(self.base_width * self.zoom_factor)
        new_height = int(self.base_height * self.zoom_factor)
        self.setFixedSize(new_width, new_height)
        self.updateGeometry()
    
    def reset_zoom(self):
        """Reset zoom to default"""
        self.zoom_factor = 1.0
        self.apply_zoom()
    
    def set_base_size(self, width, height):
        """Set the base size for zoom calculations"""
        self.base_width = width
        self.base_height = height
        self.apply_zoom()


class CSVAnalyticsDialog(QDialog): 
    def __init__(self, csv_path, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.df = None
        
        self.setWindowTitle("CSV Analytics Dashboard")
        self.setMinimumSize(1000, 700)
        
        # Detect theme and apply appropriate styling
        self.detect_and_apply_theme()
        
        self.setup_ui()
        self.load_and_analyze()

    def detect_and_apply_theme(self):
        """Detect current theme (light/dark) and apply appropriate styling"""
        # Check the application's palette to determine if dark mode is active
        palette = QApplication.instance().palette()
        bg_color = palette.color(QPalette.Window)
        
        # If background is dark (luminance < 128), use dark mode
        self.is_dark_mode = bg_color.lightness() < 128
        
        if self.is_dark_mode:
            self.apply_dark_mode_style()
            # Configure matplotlib for dark mode
            plt.style.use('dark_background')
        else:
            self.apply_light_mode_style()
            # Configure matplotlib for light mode
            plt.style.use('default')

    def load_stylesheet(self, filename):
        """Load and return stylesheet from file"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            stylesheet_path = os.path.join(current_dir, filename)
            with open(stylesheet_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Using default styling.")
            return ""
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
            return ""

    def apply_dark_mode_style(self):
        """Apply dark mode styling using the external QSS file"""
        stylesheet = self.load_stylesheet("darkMode.qss")
        if stylesheet:
            self.setStyleSheet(stylesheet)
        
        # Set theme colors for matplotlib and internal use - matching your darkmode.qss
        self.theme_colors = {
            'bg_primary': '#2D2D2D',
            'bg_secondary': '#3E3E3E',
            'bg_tertiary': '#444444',
            'border': '#444444',
            'text': '#FFFFFF',
            'text_secondary': '#CCCCCC',
            'accent': '#0078D7',
            'accent_hover': '#005A9E',
            'accent_pressed': '#004080',
            'success': '#51cf66',
            'error': '#ff6b6b',
            'alternate': '#4A4A4A',
            'figure_bg': '#2D2D2D',  # Match the main background
            'axis_bg': '#2D2D2D',    # Axis background
            'grid_color': '#444444',  # Grid lines
        }

    def apply_light_mode_style(self):
        """Apply light mode styling using the external QSS file"""
        stylesheet = self.load_stylesheet("lightMode.qss")
        if stylesheet:
            self.setStyleSheet(stylesheet)
        
        # Set theme colors for matplotlib and internal use - matching your lightmode.qss
        self.theme_colors = {
            'bg_primary': '#F5F5F5',
            'bg_secondary': '#FFFFFF',
            'bg_tertiary': '#E0E0E0',
            'border': '#CCCCCC',
            'text': '#333333',
            'text_secondary': '#666666',
            'accent': '#0078D7',
            'accent_hover': '#005A9E',
            'accent_pressed': '#004080',
            'success': '#28a745',
            'error': '#dc3545',
            'alternate': '#F9F9F9',
            'figure_bg': '#FFFFFF',
            'axis_bg': '#FFFFFF',
            'grid_color': '#CCCCCC',
        }
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        self.tab_widget = QTabWidget()

        self.overview_tab = QWidget()
        self.setup_overview_tab()
        self.tab_widget.addTab(self.overview_tab, "Overview")

        self.correlation_tab = QWidget()
        self.setup_correlation_tab()
        self.tab_widget.addTab(self.correlation_tab, "Correlations")

        self.quality_tab = QWidget()
        self.setup_quality_tab()
        self.tab_widget.addTab(self.quality_tab, "Data Quality")

        layout.addWidget(self.tab_widget)

        # Close button with improved styling
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(120)
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
    def setup_overview_tab(self):
        layout = QVBoxLayout(self.overview_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        info_group = QGroupBox("Dataset Information")
        info_layout = QFormLayout(info_group)
        info_layout.setSpacing(12)
        info_layout.setContentsMargins(20, 25, 20, 20)

        self.file_name_label = QLabel()
        self.file_size_label = QLabel()
        self.rows_label = QLabel()
        self.columns_label = QLabel()
        self.memory_usage_label = QLabel()

        # Apply styling to value labels
        for label in [self.file_name_label, self.file_size_label, self.rows_label, 
                      self.columns_label, self.memory_usage_label]:
            label.setStyleSheet("font-weight: bold;")

        info_layout.addRow("File Name:", self.file_name_label)
        info_layout.addRow("File Size:", self.file_size_label)
        info_layout.addRow("Rows:", self.rows_label)
        info_layout.addRow("Columns:", self.columns_label)
        info_layout.addRow("Memory Usage:", self.memory_usage_label)

        layout.addWidget(info_group)

        types_group = QGroupBox("Column Types")
        types_layout = QVBoxLayout(types_group)
        types_layout.setContentsMargins(15, 25, 15, 15)

        self.types_table = QTableWidget()
        self.types_table.setColumnCount(4)
        self.types_table.setHorizontalHeaderLabels(["Column", "Type", "Non-Null Count", "Unique Values"])
        self.types_table.setAlternatingRowColors(True)
        types_layout.addWidget(self.types_table)

        layout.addWidget(types_group)

        stats_group = QGroupBox("Statistical Summary")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(15, 25, 15, 15)

        self.stats_table = QTableWidget()
        self.stats_table.setAlternatingRowColors(True)
        stats_layout.addWidget(self.stats_table)

        layout.addWidget(stats_group)

    def setup_correlation_tab(self):
        layout = QVBoxLayout(self.correlation_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Control Panel
        control_container = QWidget()
        control_layout = QHBoxLayout(control_container)
        control_layout.setSpacing(20)
        control_layout.setContentsMargins(20, 15, 20, 15)
        
        # Font Size Control
        font_label = QLabel("Label Font Size:")
        control_layout.addWidget(font_label)
        
        self.font_size_combo = QComboBox()
        self.font_size_combo.addItems([str(s) for s in [6, 8, 10, 12, 14, 16, 18]])
        self.font_size_combo.setCurrentText("6")
        self.font_size_combo.currentIndexChanged.connect(self.update_correlations)
        self.font_size_combo.setFixedWidth(80)
        control_layout.addWidget(self.font_size_combo)
        
        control_layout.addSpacing(20)
        
        # Color Scheme Control
        color_label = QLabel("Color Scheme:")
        control_layout.addWidget(color_label)
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['coolwarm', 'RdBu_r', 'viridis', 'plasma', 'seismic', 'bwr'])
        self.colormap_combo.setCurrentText("coolwarm")
        self.colormap_combo.currentIndexChanged.connect(self.update_correlations)
        self.colormap_combo.setFixedWidth(120)
        control_layout.addWidget(self.colormap_combo)
        
        control_layout.addSpacing(20)
        
        # Show Values Toggle
        self.show_values_checkbox = QCheckBox("Show Values")
        self.show_values_checkbox.setChecked(False)
        self.show_values_checkbox.stateChanged.connect(self.update_correlations)
        control_layout.addWidget(self.show_values_checkbox)
        
        control_layout.addStretch()
        
        # Refresh Button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self.update_correlations)
        control_layout.addWidget(refresh_btn)
        
        # Reset Zoom Button
        reset_zoom_btn = QPushButton("Reset Zoom")
        reset_zoom_btn.setFixedWidth(120)
        reset_zoom_btn.clicked.connect(self.reset_correlation_zoom)
        control_layout.addWidget(reset_zoom_btn)

        layout.addWidget(control_container)

        # Matplotlib Figure and Canvas with Scroll Area
        self.corr_figure = Figure(figsize=(16, 13), facecolor=self.theme_colors['figure_bg'])
        self.corr_canvas = ZoomableCanvas(self.corr_figure, self)
        self.corr_canvas.setMinimumSize(800, 600)
        self.corr_canvas.set_base_size(800, 600)
        
        # Create scroll area for the canvas
        self.corr_scroll_area = QScrollArea()
        self.corr_scroll_area.setWidget(self.corr_canvas)
        self.corr_scroll_area.setWidgetResizable(False)
        self.corr_scroll_area.setAlignment(Qt.AlignCenter)
        self.corr_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.corr_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        layout.addWidget(self.corr_scroll_area, stretch=1)

        layout.addWidget(self.corr_scroll_area, stretch=1)

        # Create standard toolbar but remove all buttons except Save
        self.corr_toolbar = NavigationToolbar(self.corr_canvas, self)
        
        # Remove all buttons except Save
        buttons_to_remove = ['Home', 'Back', 'Forward', 'Pan', 'Zoom', 'Subplots','Customize']
        for action in self.corr_toolbar.actions():
            if hasattr(action, 'text') and action.text():
                action_text = action.text()
                if any(button in action_text for button in buttons_to_remove):
                    self.corr_toolbar.removeAction(action)
        
        layout.addWidget(self.corr_toolbar)
            
        # Add zoom instructions label
        zoom_info = QLabel("💡 Tip: Use mouse wheel or touchpad to zoom in/out on the correlation graph")
        zoom_info.setStyleSheet("font-size: 9pt; font-style: italic;")
        zoom_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(zoom_info)

    def reset_correlation_zoom(self):
        """Reset the correlation graph zoom to default"""
        self.corr_canvas.reset_zoom()

    def setup_quality_tab(self):
        layout = QVBoxLayout(self.quality_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Create a splitter to make the graph resizable
        splitter = QSplitter(Qt.Vertical)

        # Top widget - Missing data table
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        missing_group = QGroupBox("Missing Data Analysis")
        missing_layout = QVBoxLayout(missing_group)
        missing_layout.setContentsMargins(15, 25, 15, 15)

        self.missing_table = QTableWidget()
        self.missing_table.setColumnCount(3)
        self.missing_table.setHorizontalHeaderLabels(["Column", "Missing Count", "Missing %"])
        self.missing_table.setAlternatingRowColors(True)
        missing_layout.addWidget(self.missing_table)

        top_layout.addWidget(missing_group)

        # Bottom widget - Missing data visualization
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        viz_group = QGroupBox("Missing Data Visualization")
        viz_layout = QVBoxLayout(viz_group)
        viz_layout.setContentsMargins(15, 25, 15, 15)

        self.missing_figure = Figure(figsize=(12, 6), facecolor=self.theme_colors['figure_bg'])
        self.missing_canvas = FigureCanvas(self.missing_figure)
        viz_layout.addWidget(self.missing_canvas)

        bottom_layout.addWidget(viz_group)

        # Add widgets to splitter
        splitter.addWidget(top_widget)
        splitter.addWidget(bottom_widget)
        
        # Set initial sizes (60% for table, 40% for graph)
        splitter.setSizes([400, 300])

        layout.addWidget(splitter, stretch=1)

        # Duplicate Records section
        dup_group = QGroupBox("Duplicate Records")
        dup_layout = QFormLayout(dup_group)
        dup_layout.setSpacing(12)
        dup_layout.setContentsMargins(20, 25, 20, 20)

        self.duplicate_count_label = QLabel()
        self.duplicate_percent_label = QLabel()
        
        # Style duplicate labels
        self.duplicate_count_label.setStyleSheet("font-weight: bold;")
        self.duplicate_percent_label.setStyleSheet("font-weight: bold;")

        dup_layout.addRow("Duplicate Rows:", self.duplicate_count_label)
        dup_layout.addRow("Duplicate %:", self.duplicate_percent_label)

        layout.addWidget(dup_group)

    def load_and_analyze(self):
        try:
            try:
                self.df = pd.read_csv(self.csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    self.df = pd.read_csv(self.csv_path, encoding='latin-1')
                except UnicodeDecodeError:
                    self.df = pd.read_csv(self.csv_path, encoding='cp1252')

            if self.df.empty:
                QMessageBox.warning(self, "Warning", "The CSV file is empty.")
                return

            self.df = self.df.replace([np.inf, -np.inf], np.nan)

            self.update_overview()
            self.update_correlations()
            self.update_quality_analysis()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV file:\n{str(e)}")
    
    def update_overview(self):
        file_size = os.path.getsize(self.csv_path) / (1024 * 1024)
        self.file_name_label.setText(os.path.basename(self.csv_path))
        self.file_size_label.setText(f"{file_size:.2f} MB")
        self.rows_label.setText(f"{len(self.df):,}")
        self.columns_label.setText(f"{len(self.df.columns)}")
        self.memory_usage_label.setText(f"{self.df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")

        self.types_table.setRowCount(len(self.df.columns))
        for i, col in enumerate(self.df.columns):
            self.types_table.setItem(i, 0, QTableWidgetItem(str(col)))
            self.types_table.setItem(i, 1, QTableWidgetItem(str(self.df[col].dtype)))
            self.types_table.setItem(i, 2, QTableWidgetItem(str(self.df[col].count())))
            self.types_table.setItem(i, 3, QTableWidgetItem(str(self.df[col].nunique())))

        self.types_table.resizeColumnsToContents()

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_df = self.df[numeric_cols].describe()
            self.stats_table.setRowCount(len(stats_df))
            self.stats_table.setColumnCount(len(numeric_cols))
            self.stats_table.setHorizontalHeaderLabels(numeric_cols.tolist())
            self.stats_table.setVerticalHeaderLabels(stats_df.index.tolist())

            for i, row in enumerate(stats_df.index):
                for j, col in enumerate(numeric_cols):
                    value = stats_df.loc[row, col]
                    self.stats_table.setItem(i, j, QTableWidgetItem(f"{value:.4f}"))

            self.stats_table.resizeColumnsToContents()

    def update_correlations(self):
        self.corr_figure.clear()
        
        # Get user-selected options
        try:
            label_font_size = int(self.font_size_combo.currentText())
        except:
            label_font_size = 10
            
        colormap = self.colormap_combo.currentText()
        show_values = self.show_values_checkbox.isChecked()
        
        # Determine text colors based on theme
        text_color = self.theme_colors['text']
        title_color = self.theme_colors['accent']
        error_color = self.theme_colors['error']
        success_color = self.theme_colors['success']

        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                ax = self.corr_figure.add_subplot(111, facecolor=self.theme_colors['figure_bg'])
                ax.text(0.5, 0.5, 'Need at least 2 numeric columns for correlation',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=16, color=text_color)
                ax.set_facecolor(self.theme_colors['figure_bg'])
            else:
                # Filter for valid columns
                valid_cols = [col for col in numeric_cols if self.df[col].nunique() > 1 and not self.df[col].isna().all()]

                if len(valid_cols) < 2:
                    ax = self.corr_figure.add_subplot(111, facecolor=self.theme_colors['figure_bg'])
                    ax.text(0.5, 0.5, 'No valid columns for correlation analysis',
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, fontsize=16, color=text_color)
                    ax.set_facecolor(self.theme_colors['figure_bg'])
                else:
                    corr_matrix = self.df[valid_cols].corr()
                    ax = self.corr_figure.add_subplot(111, facecolor=self.theme_colors['figure_bg'])
                    
                    # Apply dark theme styling to the axis
                    ax.set_facecolor(self.theme_colors['axis_bg'])
                    
                    # Create heatmap
                    im = ax.imshow(corr_matrix.values, cmap=colormap, aspect='auto', vmin=-1, vmax=1)
                    
                    # Add colorbar with styling
                    cbar = self.corr_figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Correlation Coefficient', fontsize=label_font_size, color=text_color)
                    cbar.ax.tick_params(colors=text_color, labelsize=label_font_size - 2)
                    if self.is_dark_mode:
                        cbar.ax.set_facecolor(self.theme_colors['axis_bg'])
                    
                    # Set ticks and labels
                    ax.set_xticks(range(len(valid_cols)))
                    ax.set_yticks(range(len(valid_cols)))
                    ax.set_xticklabels(valid_cols, rotation=45, ha='right', fontsize=label_font_size, color=text_color)
                    ax.set_yticklabels(valid_cols, fontsize=label_font_size, color=text_color)
                    
                    # Style the spines (borders)
                    for spine in ax.spines.values():
                        spine.set_color(self.theme_colors['border'])
                        spine.set_linewidth(1)
                    
                    # Add correlation values if requested
                    if show_values:
                        for i in range(len(valid_cols)):
                            for j in range(len(valid_cols)):
                                value = corr_matrix.values[i, j]
                                # Adjust text color based on background intensity
                                if self.is_dark_mode:
                                    text_color_val = 'white' if abs(value) > 0.5 else text_color
                                else:
                                    text_color_val = 'black' if abs(value) > 0.5 else text_color
                                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                                       fontsize=max(6, label_font_size - 4), color=text_color_val, weight='bold')
                    
                    ax.set_title('Correlation Matrix', fontsize=label_font_size + 4, 
                                color=title_color, weight='bold', pad=20)
                    
                    # Grid styling
                    ax.grid(False)
                    
                    # Adjust layout
                    self.corr_figure.subplots_adjust(left=0.15, bottom=0.20, right=0.95, top=0.93)
                    
        except Exception as e:
            ax = self.corr_figure.add_subplot(111, facecolor=self.theme_colors['figure_bg'])
            ax.text(0.5, 0.5, f'Error creating correlation plot:\n{str(e)[:50]}...',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, color=error_color)
            ax.set_facecolor(self.theme_colors['figure_bg'])

        self.corr_canvas.draw()

    def update_quality_analysis(self):
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100

        self.missing_table.setRowCount(len(self.df.columns))
        for i, col in enumerate(self.df.columns):
            self.missing_table.setItem(i, 0, QTableWidgetItem(str(col)))
            self.missing_table.setItem(i, 1, QTableWidgetItem(str(missing_data[col])))
            self.missing_table.setItem(i, 2, QTableWidgetItem(f"{missing_percent[col]:.2f}%"))

        self.missing_table.resizeColumnsToContents()

        self.missing_figure.clear()
        
        # Determine text colors based on theme
        text_color = self.theme_colors['text']
        title_color = self.theme_colors['accent']
        success_color = self.theme_colors['success']
        error_color = self.theme_colors['error']
        
        try:
            if missing_data.sum() > 0:
                ax = self.missing_figure.add_subplot(111, facecolor=self.theme_colors['figure_bg'])
                missing_matrix = self.df.isnull()

                if len(self.df) > 1000:
                    sample_size = min(1000, len(self.df))
                    sample_indices = np.random.choice(len(self.df), sample_size, replace=False)
                    missing_matrix = missing_matrix.iloc[sample_indices]
                    ax.set_title(f'Missing Data Pattern (Sample of {sample_size} rows)', 
                                fontsize=14, color=title_color, weight='bold', pad=15)
                else:
                    ax.set_title('Missing Data Pattern (Red = Missing)', 
                                fontsize=14, color=title_color, weight='bold', pad=15)

                if missing_matrix.any().any():
                    im = ax.imshow(missing_matrix.T.values, cmap='Reds', aspect='auto', interpolation='nearest')
                    ax.set_yticks(range(len(self.df.columns)))
                    ax.set_yticklabels(self.df.columns, fontsize=9, color=text_color)
                    ax.set_xlabel('Rows', fontsize=11, color=text_color)
                    ax.tick_params(colors=text_color)
                    
                    # Apply dark theme styling
                    ax.set_facecolor(self.theme_colors['axis_bg'])
                    for spine in ax.spines.values():
                        spine.set_color(self.theme_colors['border'])
                    
                    cbar = self.missing_figure.colorbar(im, ax=ax)
                    cbar.set_label('Missing Values', color=text_color)
                    cbar.ax.tick_params(colors=text_color)
                    if self.is_dark_mode:
                        cbar.ax.set_facecolor(self.theme_colors['axis_bg'])
                else:
                    ax.text(0.5, 0.5, 'No missing data found',
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, fontsize=16, color=success_color, weight='bold')
                    ax.set_facecolor(self.theme_colors['figure_bg'])
            else:
                ax = self.missing_figure.add_subplot(111, facecolor=self.theme_colors['figure_bg'])
                ax.text(0.5, 0.5, 'No missing data found',
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=16, color=success_color, weight='bold')
                ax.set_facecolor(self.theme_colors['figure_bg'])
                
            ax.grid(False)
        except Exception as e:
            ax = self.missing_figure.add_subplot(111, facecolor=self.theme_colors['figure_bg'])
            ax.text(0.5, 0.5, f'Error creating missing data plot:\n{str(e)[:50]}...',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, color=error_color)
            ax.set_facecolor(self.theme_colors['figure_bg'])

        self.missing_canvas.draw()

        try:
            duplicate_count = self.df.duplicated().sum()
            duplicate_percent = (duplicate_count / len(self.df)) * 100 if len(self.df) > 0 else 0
            self.duplicate_count_label.setText(f"{duplicate_count:,}")
            self.duplicate_percent_label.setText(f"{duplicate_percent:.2f}%")
        except Exception as e:
            self.duplicate_count_label.setText("Error calculating")
            self.duplicate_percent_label.setText("Error calculating")


class MainAnalyticsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CSV Analytics Dashboard")
        self.setGeometry(100, 100, 900, 700)
        
        # Detect and apply theme
        self.detect_and_apply_theme()
        
        self.setup_ui()

    def load_stylesheet(self, filename):
        """Load and return stylesheet from file"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            stylesheet_path = os.path.join(current_dir, filename)
            with open(stylesheet_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Using default styling.")
            return ""
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
            return ""

    def detect_and_apply_theme(self):
        """Detect current theme and apply appropriate styling"""
        palette = QApplication.instance().palette()
        bg_color = palette.color(QPalette.Window)
        is_dark_mode = bg_color.lightness() < 128
        
        if is_dark_mode:
            stylesheet = self.load_stylesheet("darkMode.qss")
        else:
            stylesheet = self.load_stylesheet("lightMode.qss")
        
        if stylesheet:
            self.setStyleSheet(stylesheet)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        self.tab_widget = QTabWidget()

        self.add_csv_button = QPushButton("Add CSV File")
        self.add_csv_button.clicked.connect(self.add_csv_tab)
        self.add_csv_button.setFixedWidth(200)

        layout.addWidget(self.tab_widget)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.add_csv_button)
        layout.addLayout(button_layout)

    def add_csv_tab(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if file_path:
            csv_dialog = CSVAnalyticsDialog(file_path, self) 
            tab_index = self.tab_widget.addTab(csv_dialog, os.path.basename(file_path))
            self.tab_widget.setCurrentIndex(tab_index)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    
    # Create and show main dialog
    main_dialog = MainAnalyticsDialog()
    main_dialog.show()
    
    sys.exit(app.exec_())