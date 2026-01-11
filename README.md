C/C++ Software Defect Prediction Tool

A professional PyQt5-based desktop application designed to streamline the pipeline for software defect prediction. The tool extracts static code metrics from C/C++ source code using libclang, performs data cleaning, and labels the dataset using historical bug reports.

🚀 Key Features

Static Analysis: Deep parsing of C/C++ source code using LLVM's libclang.

Metric Extraction: Calculates LOC, Cyclomatic Complexity, Halstead measures, Maintainability Index, and more.

Automated Labeling: Efficiently matches extracted functions against bug reports to generate a binary Bug label (0 for clean, 1 for buggy).

Data Preprocessing: - Automatic removal of constant and duplicate columns.

Median-based imputation for missing numeric values.

Normalization of C++ function names (parameter and namespace stripping).

Multi-threaded UI: Long-running extraction processes run in background threads to keep the GUI responsive.

🛠️ Technical Stack

UI Framework: PyQt5

Parsing Engine: libclang (LLVM)

Data Analysis: Pandas, NumPy, Regex

Platform: Linux (Tested with LLVM Toolset 9.0)

📂 Project Structure

project_root/
├── main.py                     # Application entry point
├── core/                       # Backend Logic
│   ├── metrics_extractor.py    # libclang wrapper for extraction
│   ├── metrics_calculator.py   # Math logic for metric formulas
│   ├── extract_add_bug.py      # Integrated extraction + labeling workflow
│   └── data_preprocessing_and_labeling.py # Data cleaning & Bug labeling logic
└── ui/                         # User Interface
    ├── tabs/
    │   ├── metrics_main_tab.py # Parent container & libclang initialization
    │   └── subtabs/
    │       ├── metric_extraction/
    │       │   └── metrics_for_prediction_tab.py # Basic extraction UI
    │       └── metric_extraction_bug_labeling/
    │           └── data_preparation_tab.py       # Extraction + Labeling UI


⚙️ Setup & Installation

Prerequisites

Python 3.8+

LLVM/Clang Toolset (specifically libclang.so)

Installation

Clone the repository.

Install required dependencies:

pip install pyqt5 pandas


Ensure libclang is installed. The application is currently configured to look for the library at:
/opt/rh/llvm-toolset-9.0/root/usr/lib64/libclang.so.9
(Note: You can update this path in ui/tabs/metrics_main_tab.py if your environment differs.)

📖 Usage

Launch the App: Run python main.py.

Configure libclang: On startup, the app initializes the Clang environment.

Extraction + Labeling:

Navigate to the "Extract Metrics + Add Bug Label" tab.

Select your C/C++ source folder.

Upload your Bug Report CSV (must contain a column with function names).

Click "Process and Save".

Data Cleaning: The output CSV will be automatically cleaned (NaNs filled, duplicates removed) and ready for Machine Learning training.

🧪 Data Cleaning Logic

The tool implements a specific cleaning pipeline:

Constant Removal: Drops columns where all values are the same.

Duplicate Removal: Drops columns with different names but identical data.

Median Imputation: Fills NaN values in numeric columns with the median of that column.

Function Normalization: Simplifies Namespace::Class::Function(params) to Class::Function for robust matching.

Created for the DRDL Static Analysis Project.
