
---

# C/C++ Software Defect Prediction Application

A professional desktop application built with **PyQt5** and **LLVM/libclang** designed to automate the data engineering pipeline for software defect prediction. This tool bridges the gap between raw source code and machine-learning-ready datasets by extracting high-fidelity static metrics and correlating them with historical bug reports.

---

## 🚀 Key Features

* **Deep Static Analysis:** Leverages the production-grade **LLVM libclang** parser to navigate Abstract Syntax Trees (AST) for precise C/C++ function-level analysis.
* **Comprehensive Metric Suite:** Calculates standard industry metrics including LOC, Cyclomatic Complexity, Halstead measures, and Maintainability Index.
* **Automated Ground-Truth Labeling:** Intelligent matching engine that joins source code functions with bug report CSVs to create a binary `Bug` label.
* **Robust Data Cleaning:** A specialized pipeline that handles the "noise" of static analysis:
* **Normalization:** Strips namespaces and parameters for consistent function matching.
* **Deduplication:** Automatically identifies and removes redundant or constant features.
* **Imputation:** Strategic median-based filling for missing data points.


* **Asynchronous Processing:** Multi-threaded architecture ensures the UI remains responsive even during heavy analysis of large codebases.

---

## 🛠️ Technical Stack

| Component | Technology |
| --- | --- |
| **UI Framework** | PyQt5 (Python) |
| **Parsing Engine** | libclang (LLVM 9.0) |
| **Data Processing** | Pandas, NumPy, Regex |
| **Environment** | Linux (RHEL/CentOS/Ubuntu) |

---

## 📊 Extracted Metrics

The tool extracts a wide range of features used in defect prediction research, including:

* **Volume Metrics:** Lines of Code (LOC), Comment Density.
* **Complexity Metrics:** McCabe’s Cyclomatic Complexity .
* **Halstead Metrics:** Program Volume (), Difficulty (), and Effort ().
* **Maintainability Index (MI):** A composite metric calculated as:



---

## 📂 Project Structure

```bash
project_root/
├── main.py                     # Entry point (Initializes UI and Styles)
├── core/                       # Backend Logic & Analysis
│   ├── metrics_extractor.py    # libclang AST traversal wrapper
│   ├── metrics_calculator.py   # Mathematical formulas for metrics
│   ├── extract_add_bug.py      # Orchestrator for the integrated workflow
│   └── data_preprocessing_and_labeling.py # Cleaning & Labeling logic
└── ui/                         # User Interface Components
    ├── tabs/
    │   ├── metrics_main_tab.py # Main container & libclang configuration
    │   └── subtabs/
    │       ├── metric_extraction/      # Pure extraction interface
    │       └── metric_extraction_bug_labeling/ # Labeling & Prep interface

```

---

## ⚙️ Setup & Installation

### 1. Prerequisites

* **Python:** 3.8 or higher.
* **LLVM:** Ensure `libclang.so` is installed on your system.
* *Default Path:* `/opt/rh/llvm-toolset-9.0/root/usr/lib64/libclang.so.9`



### 2. Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/cpp-defect-predictor.git
cd cpp-defect-predictor

# Install dependencies
pip install pyqt5 pandas numpy

```

> [!IMPORTANT]
> If your `libclang.so` is located in a different directory, update the path in `ui/tabs/metrics_main_tab.py` before launching.

---

## 📖 Usage Workflow

1. **Initialize:** Run `python main.py`. The app will verify the Clang environment on startup.
2. **Configure Paths:**
* Set the **Source Folder** containing your `.cpp` or `.c` files.
* (Optional) Upload a **Bug Report CSV** containing a column of known buggy function names.


3. **Process:** Click **Process and Save**.
* The tool parses the AST  Calculates Metrics  Matches Bugs  Cleans the resulting Dataframe.


4. **Export:** The final output is a CSV optimized for training Machine Learning models (Scikit-Learn, XGBoost, etc.).

---

## 🧪 Data Cleaning Logic

The tool implements a deterministic cleaning pipeline to ensure high data quality:

* **Constant Removal:** Drops features with zero variance.
* **Function Normalization:** Transforms `Namespace::Class::Func(int)`  `Class::Func` to ensure matching reliability.
* **Imputation:** Missing values are filled using the column median to maintain distribution shape.

---
