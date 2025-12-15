#!/usr/bin/env python3
"""
This script extracts function/method definitions from a C/C++ project
using libclang and prepares the data for metric computation.
It serves as the extraction layer for the function-level bug prediction
metric pipeline.

Dependencies: clang Python bindings, pandas.
"""

import os
import re
import math
import argparse
import hashlib
import logging
import time
import pandas as pd
from .metrics_calculator import MetricsCalculator
# Import clang in a try block to handle errors gracefully
try:
    import clang.cindex
except ImportError:
    logging.error("Error: clang module not found. Please install with 'pip install clang'.")
    clang = None

# -----------------------------------------------------------
# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


try:
    import clang.cindex
except ImportError:
    logging.error("Error: clang module not found...")
    clang = None

# -----------------------------------------------------------
# NEW STANDALONE FUNCTION: Must be called ONCE before threading
def initialize_clang_library(libclang_path):
    """Initializes libclang with the specified path."""
    if clang is None:
        raise ImportError("clang module is required but not available")
        
    try:
        # Check if it has already been set to avoid errors on subsequent calls
        # Note: Some versions of libclang/bindings might still error here.
        # Calling this function once in the main thread is the best defense.
        clang.cindex.Config.set_library_file(libclang_path)
        logging.info(f"Successfully set libclang library to: {libclang_path}")
    except Exception as e:
        # If it fails, raise a helpful error
        logging.error(f"Failed to set libclang library file at {libclang_path}: {e}")
        raisee

# -----------------------------------------------------------


def initialize_clang_library(libclang_path):
    """Initializes libclang with the specified path."""
    if clang is None:
        raise ImportError("clang module is required but not available")
        
    try:
        # Check if it has already been set to avoid errors on subsequent calls
        # Note: Some versions of libclang/bindings might still error here.
        # Calling this function once in the main thread is the best defense.
        clang.cindex.Config.set_library_file(libclang_path)
        logging.info(f"Successfully set libclang library to: {libclang_path}")
    except Exception as e:
        # If it fails, raise a helpful error
        logging.error(f"Failed to set libclang library file at {libclang_path}: {e}")
        raise


class MetricsExtractor:
    """Main class for extracting function/method definitions from C/C++ files
    using libclang and managing the metric computation flow."""
    
    def __init__(self, libclang_path="/opt/rh/llvm-toolset-9.0/root/usr/lib64/libclang.so.9"):
        # Initialize the metric calculator engine
        self.calculator = MetricsCalculator()

    def initialize_libclang(self):
        """Initialize libclang with the specified path."""
        if clang is None:
            raise ImportError("clang module is required but not available")
            
        try:
            clang.cindex.Config.set_library_file(self.libclang_path)
        except Exception as e:
            logging.error(f"Failed to set libclang library file at {self.libclang_path}: {e}")
            raise

    def get_source_and_header_extensions(self):
        """Returns sets of source and header file extensions."""
        source_exts = {'.c', '.cpp', '.cc', '.cxx', '.cp'}
        header_exts = {'.h', '.hxx', '.hp', '.hpp'}
        return source_exts, header_exts

    def collect_include_paths(self, root_folder, header_exts):
        """Recursively collect directories that contain header files."""
        include_paths = set()
        for dirpath, _, files in os.walk(root_folder):
            for f in files:
                _, ext = os.path.splitext(f)
                if ext.lower() in header_exts:
                    include_paths.add(os.path.abspath(dirpath))
                    break
        return list(include_paths)

    def get_function_signature(self, node, file_path):
        """Extracts the first line (signature) from the function definition."""
        if node.extent:
            start = node.extent.start
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if 0 < start.line <= len(lines):
                        return lines[start.line - 1].strip()
            except Exception as e:
                logging.warning(f"Could not read {file_path}: {e}")
        return ""

    def get_full_code(self, node, file_path):
        """Extracts the complete function/method code from the file."""
        if node.extent:
            start = node.extent.start
            end = node.extent.end
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    return "".join(lines[start.line - 1:end.line]).strip()
            except Exception as e:
                logging.warning(f"Could not read {file_path}: {e}")
        return ""

    def unique_function_key(self, node, file_path):
        """Generates a unique key based on file path and starting location."""
        if node.extent:
            start = node.extent.start
            key_str = f"{os.path.abspath(file_path)}:{start.line}:{start.column}:{node.spelling}"
            return hashlib.md5(key_str.encode("utf-8")).hexdigest()
        return None

    def visit_node(self, node, file_path, extracted_functions):
        """Recursively visits AST nodes to extract function definitions."""
        valid_kinds = {
            clang.cindex.CursorKind.FUNCTION_DECL,
            clang.cindex.CursorKind.CXX_METHOD,
            clang.cindex.CursorKind.CONSTRUCTOR,
            clang.cindex.CursorKind.DESTRUCTOR,
            clang.cindex.CursorKind.CONVERSION_FUNCTION,
            clang.cindex.CursorKind.FUNCTION_TEMPLATE,
            clang.cindex.CursorKind.FRIEND_DECL,
        }
        if node.kind in valid_kinds:
            if node.is_definition() and node.location.file and \
                os.path.abspath(node.location.file.name) == os.path.abspath(file_path):
                key = self.unique_function_key(node, file_path)
                if key not in extracted_functions:
                    fName = node.spelling
                    if node.semantic_parent and node.semantic_parent.kind in {
                            clang.cindex.CursorKind.STRUCT_DECL,
                            clang.cindex.CursorKind.CLASS_DECL,
                            clang.cindex.CursorKind.CLASS_TEMPLATE
                    }:
                        fName = f"{node.semantic_parent.spelling}::{fName}"
                    fSignature = self.get_function_signature(node, file_path)
                    fCode = self.get_full_code(node, file_path)
                    fPath = os.path.abspath(file_path).replace("\\", "/")
                    extracted_functions[key] = {
                        "Location": fPath,
                        "Function": fName,
                        "fSignature": fSignature,
                        "fCode": fCode,
                    }
        for child in node.get_children():
            if child.location and child.location.file:
                if not child.location.file.name.startswith("<"):
                    self.visit_node(child, file_path, extracted_functions)

    def parse_file(self, file_path, clang_args, extracted_functions):
        """Parses a file and extracts function definitions."""
        index = clang.cindex.Index.create()
        try:
            # Note: The original code does not handle the return of the TU,
            # but relies on the side effect of visit_node modifying extracted_functions.
            translation_unit = index.parse(file_path, args=clang_args)
            self.visit_node(translation_unit.cursor, file_path, extracted_functions)
        except Exception as e:
            logging.error(f"Error parsing {file_path}: {e}")

    def extract_functions_from_folder(self, folder_path, progress_callback=None):
        """
        Extracts function definitions from all supported files in the folder.
        """
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Directory not found: {folder_path}")
            
        source_exts, header_exts = self.get_source_and_header_extensions()
        include_paths = self.collect_include_paths(folder_path, header_exts)
        clang_args = [f"-I{p}" for p in include_paths]
        
        extracted_functions = {}
        f = 0
        total_files = 0
        processed_files = 0
        
        # First count total files for progress reporting
        for root, _, files in os.walk(folder_path):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in source_exts.union(header_exts):
                    total_files += 1
        
        # Then process each file
        for root, _, files in os.walk(folder_path):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in source_exts.union(header_exts):
                    file_path = os.path.join(root, file)
                    self.parse_file(file_path, clang_args, extracted_functions)
                    f = f + 1
                    processed_files += 1
                    
                    # Report progress if callback provided
                    if progress_callback and total_files > 0:
                        # Report 0-50% for extraction
                        progress_callback(processed_files / total_files * 50)
                        
                    if f == 100:
                        time.sleep(0.2)
                        f = 0
                        
        return list(extracted_functions.values())

    def process_folder(self, folder_path, output_csv_path, progress_callback=None):
        """
        Process a folder, compute metrics, and save results to a CSV file.
        """
        
        # 1. Extraction (0-50% progress)
        functions_list = self.extract_functions_from_folder(
            folder_path, 
            lambda p: progress_callback(p) if progress_callback else None
        )
        
        if not functions_list:
            logging.error("No functions found.")
            return None
        
        # 2. Metric Computation (50-100% progress)
        total_funcs = len(functions_list)
        all_metrics = []
        for i, func in enumerate(functions_list):
            code = func["fCode"]
            signature = func["fSignature"]
            
            # Delegation to the dedicated calculator class
            metrics = self.calculator.compute_function_metrics(code, signature)
            
            # Combine function metadata with metrics
            func_metrics = {**func, **metrics}
            all_metrics.append(func_metrics)
            
            # Report progress if callback provided (start from 50)
            if progress_callback:
                computation_progress = (i + 1) / total_funcs * 50
                progress_callback(50 + computation_progress)
        
        # 3. Save to CSV
        df = pd.DataFrame(all_metrics)
        try:
            # Use os.path.abspath to ensure the path is fully resolved before opening
            if output_csv_path:
                abs_csv_path = os.path.abspath(output_csv_path)
                # The original logic saved to a temp_file wrapper which is slightly odd,
                # refactoring to a direct save which is the practical intent.
                df.to_csv(abs_csv_path, index=False)
                logging.info(f"Extraction and metric computation completed. Results saved to {abs_csv_path}")
            else:
                logging.info(f"Extraction and metric computation completed. Results Not being saved!!")
            return df
        except Exception as e:
            logging.error(f"Error writing to CSV: {e}")
            return None
