#!/usr/bin/env python3
import os
import sys
import re
import math
import argparse
import hashlib
import logging
import time
import tempfile
import atexit
import pandas as pd

try:
    import clang.cindex
except ImportError:
    logging.error("Error: clang module not found. Please install with 'pip install clang'.")
    clang = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class MetricsExtractor:
    def __init__(self, libclang_path="/usr/lib64/libclang.so"):
        self.libclang_path = libclang_path
        self.initialize_libclang()

    def initialize_libclang(self):
        if clang is None:
            raise ImportError("clang module is required but not available")
        try:
            clang.cindex.Config.set_library_file(self.libclang_path)
        except Exception as e:
            logging.error(f"Failed to set libclang library file at {self.libclang_path}: {e}")
            raise

    def get_source_and_header_extensions(self):
        source_exts = {'.c', '.cpp', '.cc', '.cxx', '.cp', '.c++'}
        header_exts = {'.h', '.hxx', '.hp', '.hpp', '.h++', '.hh', '.inl'}
        return source_exts, header_exts

    def collect_include_paths(self, root_folder, header_exts):
        include_paths = set()
        for dirpath, _, files in os.walk(root_folder):
            for f in files:
                _, ext = os.path.splitext(f)
                if ext.lower() in header_exts:
                    include_paths.add(os.path.abspath(dirpath))
                    break
        return list(include_paths)

    def get_function_signature(self, node, file_path):
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
        if node.extent:
            start = node.extent.start
            end = node.extent.end
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    # Get more lines to ensure we capture the complete function
                    # Sometimes clang's extent might not capture everything
                    start_line = max(0, start.line - 3)  # Start a bit earlier
                    end_line = min(len(lines), end.line + 3)  # End a bit later
                    return "".join(lines[start_line:end_line]).strip()
            except Exception as e:
                logging.warning(f"Could not read {file_path}: {e}")
        return ""

    def unique_function_key(self, node, file_path):
        if node.extent:
            start = node.extent.start
            key_str = f"{os.path.abspath(file_path)}:{start.line}:{start.column}:{node.spelling}"
            return hashlib.md5(key_str.encode("utf-8")).hexdigest()
        return None

    def visit_node(self, node, file_path, extracted_functions):
        valid_kinds = {
            clang.cindex.CursorKind.FUNCTION_DECL,
            clang.cindex.CursorKind.CXX_METHOD,
            clang.cindex.CursorKind.CONSTRUCTOR,
            clang.cindex.CursorKind.DESTRUCTOR,
            clang.cindex.CursorKind.CONVERSION_FUNCTION,
            clang.cindex.CursorKind.FUNCTION_TEMPLATE,
            clang.cindex.CursorKind.FRIEND_DECL,
        }
        
        # Try to extract function regardless of whether it's a definition or not
        # This will catch declarations in headers too
        if node.kind in valid_kinds:
            try:
                location = node.location
                if location and location.file:
                    actual_path = os.path.abspath(location.file.name)
                    target_path = os.path.abspath(file_path)
                    
                    # Allow parsing even if paths don't match exactly (for includes)
                    # But make sure it's from the same file system location
                    if actual_path == target_path or file_path in actual_path or actual_path in file_path:
                        key = self.unique_function_key(node, file_path)
                        if key and key not in extracted_functions:
                            fName = node.spelling
                            
                            # Build full qualified name with namespaces/classes
                            parent_cursor = node.semantic_parent
                            full_name_parts = []
                            while parent_cursor:
                                if parent_cursor.kind in {
                                    clang.cindex.CursorKind.NAMESPACE,
                                    clang.cindex.CursorKind.CLASS_DECL,
                                    clang.cindex.CursorKind.STRUCT_DECL,
                                    clang.cindex.CursorKind.CLASS_TEMPLATE
                                }:
                                    if parent_cursor.spelling:
                                        full_name_parts.append(parent_cursor.spelling)
                                parent_cursor = parent_cursor.semantic_parent
                            
                            if full_name_parts:
                                full_name_parts.reverse()
                                fName = "::".join(full_name_parts) + "::" + fName
                            
                            fSignature = self.get_function_signature(node, file_path)
                            fCode = self.get_full_code(node, file_path)
                            fPath = os.path.abspath(file_path).replace("\\", "/")
                            
                            extracted_functions[key] = {
                                "Location": fPath,
                                "Function": fName,
                                "fSignature": fSignature,
                                "fCode": fCode,
                            }
            except Exception as e:
                logging.debug(f"Error processing node {node.spelling}: {e}")
        
        # Recursively visit children
        try:
            for child in node.get_children():
                self.visit_node(child, file_path, extracted_functions)
        except Exception as e:
            logging.debug(f"Error iterating children: {e}")

    def parse_file_with_fallback(self, file_path, clang_args, extracted_functions):
        """Try multiple parsing strategies to get all functions"""
        strategies = [
            self.parse_with_default_args,
            self.parse_with_permissive_args,
            self.parse_as_cpp_wrapper,
        ]
        
        for strategy in strategies:
            try:
                strategy(file_path, clang_args, extracted_functions)
                # If we got at least one function, consider it successful
                if extracted_functions:
                    return True
            except Exception as e:
                logging.debug(f"Strategy {strategy.__name__} failed for {file_path}: {e}")
                continue
        
        return False

    def parse_with_default_args(self, file_path, clang_args, extracted_functions):
        """Original parsing strategy"""
        index = clang.cindex.Index.create()
        
        # Determine language based on extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() in {'.c'}:
            lang_args = ['-x', 'c', '-std=c11']
        else:
            lang_args = ['-x', 'c++', '-std=c++11']
        
        all_args = clang_args + lang_args + [
            '-Wno-everything',
            '-ferror-limit=0',
        ]
        
        try:
            translation_unit = index.parse(file_path, args=all_args)
            if translation_unit:
                self.visit_node(translation_unit.cursor, file_path, extracted_functions)
        except Exception as e:
            raise Exception(f"Default parse failed: {e}")

    def parse_with_permissive_args(self, file_path, clang_args, extracted_functions):
        """More permissive parsing"""
        index = clang.cindex.Index.create()
        
        all_args = clang_args + [
            '-x', 'c++',
            '-std=c++11',
            '-Wno-everything',
            '-ferror-limit=0',
            '-D__cplusplus',
            '-D__GNUC__',
            '-D__GNUG__',
            '-D__linux__',
        ]
        
        try:
            translation_unit = index.parse(
                file_path,
                args=all_args,
                options=clang.cindex.TranslationUnit.PARSE_INCOMPLETE |
                       clang.cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES
            )
            if translation_unit:
                self.visit_node(translation_unit.cursor, file_path, extracted_functions)
        except Exception as e:
            raise Exception(f"Permissive parse failed: {e}")

    def parse_as_cpp_wrapper(self, file_path, clang_args, extracted_functions):
        """Treat all files as C++ with wrapper"""
        index = clang.cindex.Index.create()
        
        # Create a temporary wrapper for problematic files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as tmp:
            tmp.write(f"""
// Wrapper for parsing {file_path}
#ifdef __cplusplus
extern "C" {{
#endif
#include "{file_path}"
#ifdef __cplusplus
}}
#endif
""")
            tmp_path = tmp.name
        
        try:
            all_args = clang_args + [
                '-x', 'c++',
                '-std=c++11',
                '-Wno-everything',
                '-ferror-limit=0',
                '-D__cplusplus',
            ]
            
            translation_unit = index.parse(tmp_path, args=all_args)
            if translation_unit:
                # When using wrapper, we still want to attribute functions to original file
                original_extracted = {}
                self.visit_node(translation_unit.cursor, tmp_path, original_extracted)
                
                # Transfer functions to original file's collection
                for key, func in original_extracted.items():
                    # Update location to original file
                    func["Location"] = os.path.abspath(file_path).replace("\\", "/")
                    extracted_functions[key] = func
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    def parse_file(self, file_path, clang_args, extracted_functions):
        """Main parsing entry point - tries multiple strategies"""
        try:
            self.parse_file_with_fallback(file_path, clang_args, extracted_functions)
        except Exception as e:
            logging.error(f"All parsing strategies failed for {file_path}: {e}")

    def extract_functions_from_folder(self, folder_path, progress_callback=None):
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Directory not found: {folder_path}")
        
        source_exts, header_exts = self.get_source_and_header_extensions()
        include_paths = self.collect_include_paths(folder_path, header_exts)
        clang_args = [f"-I{p}" for p in include_paths]
        
        # Add some common include paths for better parsing
        common_includes = [
            '/usr/include',
            '/usr/local/include',
        ]
        for inc in common_includes:
            if os.path.exists(inc):
                clang_args.append(f"-I{inc}")
        
        extracted_functions = {}
        total_files = 0
        processed_files = 0
        
        # Count total files
        for _, _, files in os.walk(folder_path):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in source_exts.union(header_exts):
                    total_files += 1
        
        # Parse files
        for root, _, files in os.walk(folder_path):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in source_exts.union(header_exts):
                    file_path = os.path.join(root, file)
                    
                    # Create a fresh dictionary for this file
                    file_functions = {}
                    
                    logging.debug(f"Parsing: {file_path}")
                    self.parse_file(file_path, clang_args, file_functions)
                    
                    # Merge results
                    extracted_functions.update(file_functions)
                    
                    processed_files += 1
                    if progress_callback and total_files > 0:
                        progress = (processed_files / total_files) * 100
                        progress_callback(progress)
                    
                    # Small delay for UI responsiveness
                    if processed_files % 20 == 0:
                        time.sleep(0.05)
        
        return list(extracted_functions.values())

    def compute_loc_metrics(self, lines):
        loc_total = len(lines)
        loc_blank = sum(1 for line in lines if line.strip() == "")
        in_block = False
        loc_comments = 0
        for line in lines:
            stripped = line.strip()
            if in_block:
                loc_comments += 1
                if "*/" in stripped:
                    in_block = False
                continue
            if stripped.startswith("//"):
                loc_comments += 1
            elif "/*" in stripped:
                loc_comments += 1
                if "*/" not in stripped:
                    in_block = True
        loc_executable = loc_total - loc_blank - loc_comments
        return loc_total, loc_blank, loc_comments, loc_executable

    def tokenize_code(self, code):
        return re.findall(r'\w+|[^\s\w]', code)

    def compute_halstead_metrics(self, code):
        operators_set = {
            "+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">=",
            "&&", "||", "!", "&", "|", "^", "~", "<<", ">>", "++", "--",
            "+=", "-=", "*=", "/=", "%="
        }
        tokens = self.tokenize_code(code)
        total_operators = 0
        total_operands = 0
        unique_operators = set()
        unique_operands = set()
        for token in tokens:
            if token in operators_set:
                total_operators += 1
                unique_operators.add(token)
            elif re.match(r'\w+', token):
                total_operands += 1
                unique_operands.add(token)
        halstead_length = total_operators + total_operands
        halstead_vocabulary = len(unique_operators) + len(unique_operands)
        halstead_volume = halstead_length * math.log2(halstead_vocabulary) if halstead_vocabulary > 0 else 0
        halstead_difficulty = (len(unique_operators) / 2) * (total_operands / len(unique_operands)) if len(unique_operands) > 0 else 0
        halstead_effort = halstead_difficulty * halstead_volume
        halstead_content = halstead_volume / 3000.0 if 3000 else 0
        halstead_prog_time = halstead_effort / 18.0
        halstead_error_est = (halstead_volume ** (2/3)) / 3000.0 if halstead_volume > 0 else 0
        halstead_level = 1 / halstead_difficulty if halstead_difficulty != 0 else 0
        return {
            "HALSTEAD_LENGTH": halstead_length,
            "HALSTEAD_VOCABULARY": halstead_vocabulary,
            "HALSTEAD_VOLUME": halstead_volume,
            "HALSTEAD_DIFFICULTY": halstead_difficulty,
            "HALSTEAD_EFFORT": halstead_effort,
            "HALSTEAD_CONTENT": halstead_content,
            "HALSTEAD_PROG_TIME": halstead_prog_time,
            "HALSTEAD_ERROR_EST": halstead_error_est,
            "HALSTEAD_LEVEL": halstead_level,
        }

    def compute_decision_metrics(self, code, loc_executable):
        decision_keywords = re.findall(r'\b(if|for|while|case|catch)\b', code)
        logical_ops = re.findall(r'&&|\|\|', code)
        decision_count = len(decision_keywords) + len(logical_ops)
        decision_density = decision_count / loc_executable if loc_executable > 0 else 0
        return decision_count, decision_density

    def compute_cfg_metrics(self, decision_count):
        cyclomatic_complexity = decision_count + 1
        node_count = cyclomatic_complexity + 1
        edge_count = node_count + decision_count - 1
        return cyclomatic_complexity, node_count, edge_count

    def compute_function_metrics(self, code, signature):
        lines = code.splitlines()
        loc_total, loc_blank, loc_comments, loc_executable = self.compute_loc_metrics(lines)
        number_of_lines = loc_total
        decision_count, decision_density = self.compute_decision_metrics(code, loc_executable)
        cyclomatic_complexity, node_count, edge_count = self.compute_cfg_metrics(decision_count)
        essential_complexity = cyclomatic_complexity
        essential_density = essential_complexity / loc_executable if loc_executable > 0 else 0
        design_complexity = cyclomatic_complexity
        design_density = design_complexity / loc_executable if loc_executable > 0 else 0
        halstead = self.compute_halstead_metrics(code)
        call_pairs = len(re.findall(r'\b\w+\s*\(', code))
        param_match = re.search(r'\((.*)\)', signature)
        if param_match:
            params = param_match.group(1).strip()
            if params == "" or params == "void":
                parameter_count = 0
            else:
                parameter_count = len([p for p in params.split(',') if p.strip() != ""])
        else:
            parameter_count = 0
        branch_count = len(re.findall(r'\bcase\b', code))
        condition_count = len(re.findall(r'(<=|>=|==|!=|<|>)', code))
        modified_condition_count = len(re.findall(r'\bif\s*\(.*?=.*?\)', code))
        multiple_condition_count = 0
        for line in lines:
            if line.strip().startswith("if"):
                if len(re.findall(r'&&|\|\|', line)) > 1:
                    multiple_condition_count += 1
        fan_in_direct = 0
        fan_in_transitive = 0
        fan_out_direct = call_pairs
        fan_out_transitive = fan_out_direct
        depth_of_call_tree = 1
        var_decls = re.findall(r'\b(int|float|double|char|long|short)\b\s+\w+', code)
        stack_size_function = len(var_decls) * 4
        stack_size_aggregate = stack_size_function
        knots = len(re.findall(r'\bgoto\b', code))
        max_essential_knots = knots
        min_essential_knots = knots
        number_of_returns = len(re.findall(r'\breturn\b', code))
        count_decl_method = 1 if signature else 0
        count_decl_method_const = 1 if signature.strip().endswith("const") else 0
        count_decl_method_friend = 1 if "friend" in code else 0
        count_decl_instance_method = 1 if ("::" in signature and "static" not in signature) else 0
        count_input = len(re.findall(r'\b(scanf|cin)\b', code))
        count_output = len(re.findall(r'\b(printf|cout)\b', code))
        count_semicolon = code.count(';')
        code_smells_count = 0
        if loc_total > 100:
            code_smells_count += 1
        if parameter_count > 5:
            code_smells_count += 1
        inheritance_depth = 1 if "::" in signature else 0
        coupling_between_objects = fan_out_direct
        lack_of_cohesion_of_methods = 0
        weighted_methods_per_class = cyclomatic_complexity
        try:
            mi = 171 - 5.2 * math.log(halstead["HALSTEAD_VOLUME"]) - 0.23 * cyclomatic_complexity - 16.2 * math.log(loc_executable)
        except ValueError:
            mi = 0
        maintainability_index = mi
        refactorability_index = maintainability_index
        cognitive_complexity = cyclomatic_complexity
        nonblank_lines = [line.strip() for line in lines if line.strip() != ""]
        if nonblank_lines:
            duplicate_count = len(nonblank_lines) - len(set(nonblank_lines))
            code_duplication_ratio = duplicate_count / len(nonblank_lines)
        else:
            code_duplication_ratio = 0
        static_code_warnings = 0
        metrics = {
            "LOC_BLANK": loc_blank,
            "LOC_COMMENTS": loc_comments,
            "LOC_EXECUTABLE": loc_executable,
            "LOC_TOTAL": loc_total,
            "NUMBER_OF_LINES": number_of_lines,
            "DECISION_COUNT": decision_count,
            "DECISION_DENSITY": decision_density,
            "CYCLOMATIC_COMPLEXITY": cyclomatic_complexity,
            "ESSENTIAL_COMPLEXITY": essential_complexity,
            "ESSENTIAL_DENSITY": essential_density,
            "DESIGN_COMPLEXITY": design_complexity,
            "DESIGN_DENSITY": design_density,
            "HALSTEAD_LENGTH": halstead["HALSTEAD_LENGTH"],
            "HALSTEAD_VOCABULARY": halstead["HALSTEAD_VOCABULARY"],
            "HALSTEAD_VOLUME": halstead["HALSTEAD_VOLUME"],
            "HALSTEAD_DIFFICULTY": halstead["HALSTEAD_DIFFICULTY"],
            "HALSTEAD_EFFORT": halstead["HALSTEAD_EFFORT"],
            "HALSTEAD_CONTENT": halstead["HALSTEAD_CONTENT"],
            "HALSTEAD_PROG_TIME": halstead["HALSTEAD_PROG_TIME"],
            "HALSTEAD_ERROR_EST": halstead["HALSTEAD_ERROR_EST"],
            "HALSTEAD_LEVEL": halstead["HALSTEAD_LEVEL"],
            "CALL_PAIRS": call_pairs,
            "PARAMETER_COUNT": parameter_count,
            "BRANCH_COUNT": branch_count,
            "EDGE_COUNT": edge_count,
            "NODE_COUNT": node_count,
            "CONDITION_COUNT": condition_count,
            "MODIFIED_CONDITION_COUNT": modified_condition_count,
            "MULTIPLE_CONDITION_COUNT": multiple_condition_count,
            "FAN_IN_DIRECT": fan_in_direct,
            "FAN_IN_TRANSITIVE": fan_in_transitive,
            "FAN_OUT_DIRECT": fan_out_direct,
            "FAN_OUT_TRANSITIVE": fan_out_transitive,
            "DEPTH_OF_CALL_TREE": depth_of_call_tree,
            "STACK_SIZE_FUNCTION": stack_size_function,
            "STACK_SIZE_AGGREGATE": stack_size_aggregate,
            "KNOTS": knots,
            "MAX_ESSENTIAL_KNOTS": max_essential_knots,
            "MIN_ESSENTIAL_KNOTS": min_essential_knots,
            "NUMBER_OF_RETURNS": number_of_returns,
            "COUNT_DECL_METHOD": count_decl_method,
            "COUNT_DECL_METHOD_CONST": count_decl_method_const,
            "COUNT_DECL_METHOD_FRIEND": count_decl_method_friend,
            "COUNT_DECL_INSTANCE_METHOD": count_decl_instance_method,
            "COUNT_INPUT": count_input,
            "COUNT_OUTPUT": count_output,
            "COUNT_SEMICOLON": count_semicolon,
            "CODE_SMELLS_COUNT": code_smells_count,
            "INHERITANCE_DEPTH": inheritance_depth,
            "COUPLING_BETWEEN_OBJECTS": coupling_between_objects,
            "LACK_OF_COHESION_OF_METHODS": lack_of_cohesion_of_methods,
            "WEIGHTED_METHODS_PER_CLASS": weighted_methods_per_class,
            "REFACTORABILITY_INDEX": refactorability_index,
            "MAINTAINABILITY_INDEX": maintainability_index,
            "COGNITIVE_COMPLEXITY": cognitive_complexity,
            "CODE_DUPLICATION_RATIO": code_duplication_ratio,
            "STATIC_CODE_WARNINGS": static_code_warnings,
        }
        return metrics

    def process_folder(self, folder_path, output_csv_path, progress_callback=None):
        functions_list = self.extract_functions_from_folder(folder_path, progress_callback)
        if not functions_list:
            logging.error("No functions found.")
            return None
        total_funcs = len(functions_list)
        all_metrics = []
        for i, func in enumerate(functions_list):
            code = func["fCode"]
            signature = func["fSignature"]
            metrics = self.compute_function_metrics(code, signature)
            func_metrics = {**func, **metrics}
            all_metrics.append(func_metrics)
            if progress_callback:
                progress_callback((i + 1) / total_funcs * 100)
        df = pd.DataFrame(all_metrics)
        try:
            with open(output_csv_path, 'w', encoding='utf-8') as temp_file:
                df.to_csv(temp_file, index=False)
            logging.info(f"Extraction completed. Found {len(functions_list)} functions.")
            return df
        except Exception as e:
            logging.error(f"Error writing to CSV: {e}")
            return None

def main(folder_path, output_csv_path):
    source_folder = folder_path
    output_csv = output_csv_path
    libclang_path = "/opt/rh/llvm-toolset-9.0/root/usr/lib64/libclang.so.9"
    try:
        extractor = MetricsExtractor(libclang_path)
        extractor.process_folder(source_folder, output_csv)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        sys.exit(1)
