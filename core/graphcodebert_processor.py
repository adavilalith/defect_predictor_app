"""
This module prepares function-level data for GraphCodeBERT.
It focuses on tokenization and extracting data flow edges (DFG)
to leverage the structural properties of the code.
"""

import re
import logging
import json

class GraphCodeBERTProcessor:
    def __init__(self, max_seq_length=512, max_nodes=64):
        self.max_seq_length = max_seq_length
        self.max_nodes = max_nodes
        # Simple keywords for C/C++ to help isolate variable identifiers
        self.cpp_keywords = {
            'int', 'float', 'double', 'char', 'long', 'void', 'if', 'else', 
            'for', 'while', 'return', 'class', 'struct', 'public', 'private'
        }

    def clean_code(self, code):
        """Removes comments to prevent them from being tokenized as code."""
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code.strip()

    def get_identifiers(self, code):
        """
        Extracts variable names and identifiers.
        In a full implementation, this uses the DFG (Data Flow Graph).
        Here, we isolate tokens that are likely variable nodes.
        """
        tokens = re.findall(r'\b[a-zA-Z_]\w*\b', code)
        identifiers = [t for t in tokens if t not in self.cpp_keywords]
        # De-duplicate while preserving order
        seen = set()
        unique_ids = [x for x in identifiers if not (x in seen or seen.add(x))]
        return unique_ids[:self.max_nodes]

    def generate_data_flow(self, code, identifiers):
        """
        Generates a simplified Data Flow Graph (DFG) mapping.
        GraphCodeBERT expects: (variable_name, index, [list_of_indices_it_depends_on])
        """
        dfg_edges = []
        for i, var in enumerate(identifiers):
            # Simplified logic: find where this variable appears after its first definition
            # In a production GraphCodeBERT setup, you'd use a tree-sitter or Clang CFG here.
            dependencies = []
            if i > 0:
                # Naive assumption: variable depends on the previous identifier in the stream
                dependencies.append(i - 1)
            dfg_edges.append((var, i, dependencies))
        return dfg_edges

    def prepare_json_line(self, function_data):
        """
        Formats the extracted function into the JSONL format 
        standardized by the CodeXGLUE / GraphCodeBERT authors.
        """
        code = self.clean_code(function_data.get("fCode", ""))
        identifiers = self.get_identifiers(code)
        dfg = self.generate_data_flow(code, identifiers)

        # This dictionary mirrors the structure required for GraphCodeBERT fine-tuning
        output = {
            "func": function_data.get("Function", "unknown"),
            "path": function_data.get("Location", "unknown"),
            "code": code,
            "tokens": code.split(), # Simplified tokenization
            "identifiers": identifiers,
            "dfg": dfg
        }
        
        return output

    def process_for_inference(self, functions_list):
        """
        Converts a list of functions from the Extractor into a 
        GraphCodeBERT-ready dataset.
        """
        processed_data = []
        for func in functions_list:
            try:
                processed_data.append(self.prepare_json_line(func))
            except Exception as e:
                logging.error(f"Failed to process function for GraphCodeBERT: {e}")
        
        return processed_data