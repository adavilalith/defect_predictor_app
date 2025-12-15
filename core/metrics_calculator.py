"""
This module contains the logic for calculating all 57 software metrics
at the function level. This logic is separated from the AST parsing
and I/O operations for better modularity.

Note: The internal logic strictly adheres to the original script's 
approximations and calculations.
"""
import re
import math
import logging

class MetricsCalculator:
    """Encapsulates all logic for computing software metrics based on raw code."""

    def __init__(self):
        # Configuration or constants can be stored here if needed later
        pass

    # --- Original Metric Computation Functions (LOGIC UNCHANGED) ---

    def compute_loc_metrics(self, lines):
        """
        Compute LOC_TOTAL, LOC_BLANK, LOC_COMMENTS, and LOC_EXECUTABLE.
        """
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
        """
        Tokenize the code into tokens for Halstead metric calculations.
        Uses the original simple regex.
        """
        return re.findall(r'\w+|[^\s\w]', code)

    def compute_halstead_metrics(self, code):
        """
        Compute Halstead metrics (LOGIC UNCHANGED).
        """
        operators_set = {"+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">=", 
                         "&&", "||", "!", "&", "|", "^", "~", "<<", ">>", "++", "--",
                         "+=", "-=", "*=", "/=", "%="}
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
        
        # Original logic had an explicit check for len(unique_operands) > 0
        unique_operands_count = len(unique_operands)
        halstead_difficulty = (len(unique_operators) / 2) * (total_operands / unique_operands_count) if unique_operands_count > 0 else 0
        
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
        """
        Compute decision count and density (LOGIC UNCHANGED).
        """
        decision_keywords = re.findall(r'\b(if|for|while|case|catch)\b', code)
        logical_ops = re.findall(r'&&|\|\|', code)
        decision_count = len(decision_keywords) + len(logical_ops)
        decision_density = decision_count / loc_executable if loc_executable > 0 else 0
        return decision_count, decision_density

    def compute_cfg_metrics(self, decision_count):
        """
        Approximate Cyclomatic Complexity and related CFG metrics (LOGIC UNCHANGED).
        """
        cyclomatic_complexity = decision_count + 1
        node_count = cyclomatic_complexity + 1
        edge_count = node_count + decision_count - 1
        return cyclomatic_complexity, node_count, edge_count

    def compute_function_metrics(self, code, signature):
        """
        Given function code and signature, compute all 57 metrics (LOGIC UNCHANGED).
        """
        lines = code.splitlines()
        (loc_total, loc_blank, loc_comments, loc_executable) = self.compute_loc_metrics(lines)
        number_of_lines = loc_total  # alias

        # Decision metrics
        decision_count, decision_density = self.compute_decision_metrics(code, loc_executable)
        cyclomatic_complexity, node_count, edge_count = self.compute_cfg_metrics(decision_count)

        # For simplicity, essential and design complexities equal cyclomatic complexity.
        essential_complexity = cyclomatic_complexity
        essential_density = essential_complexity / loc_executable if loc_executable > 0 else 0
        design_complexity = cyclomatic_complexity
        design_density = design_complexity / loc_executable if loc_executable > 0 else 0

        # Halstead metrics
        halstead = self.compute_halstead_metrics(code)

        # Call pairs: approximate using regex
        call_pairs = len(re.findall(r'\b\w+\s*\(', code))

        # Parameter count from function signature
        param_match = re.search(r'\((.*)\)', signature)
        if param_match:
            params = param_match.group(1).strip()
            if params == "" or params == "void":
                parameter_count = 0
            else:
                parameter_count = len([p for p in params.split(',') if p.strip() != ""])
        else:
            parameter_count = 0

        # Branch count: count occurrences of 'case'
        branch_count = len(re.findall(r'\bcase\b', code))
        # Condition count: relational operators in decisions
        condition_count = len(re.findall(r'(<=|>=|==|!=|<|>)', code))
        # Modified conditions: look for assignment within if condition
        modified_condition_count = len(re.findall(r'\bif\s*\(.*?=.*?\)', code))
        # Multiple conditions: if a condition contains more than one && or ||
        multiple_condition_count = 0
        for line in lines:
            if line.strip().startswith("if"):
                if len(re.findall(r'&&|\|\|', line)) > 1:
                    multiple_condition_count += 1

        # Fan in/out and call tree depth (global analysis required) – approximated here.
        fan_in_direct = 0
        fan_in_transitive = 0
        fan_out_direct = call_pairs
        fan_out_transitive = fan_out_direct
        depth_of_call_tree = 1

        # Stack size: approximate by counting common local variable declarations.
        var_decls = re.findall(r'\b(int|float|double|char|long|short)\b\s+\w+', code)
        stack_size_function = len(var_decls) * 4  # assume 4 bytes per variable
        stack_size_aggregate = stack_size_function

        # KNOTS: count goto statements.
        knots = len(re.findall(r'\bgoto\b', code))
        max_essential_knots = knots
        min_essential_knots = knots

        # Number of returns.
        number_of_returns = len(re.findall(r'\breturn\b', code))

        # Method declaration counts (at function level, these are based on the signature)
        count_decl_method = 1 if signature else 0
        count_decl_method_const = 1 if signature.strip().endswith("const") else 0
        count_decl_method_friend = 1 if "friend" in code else 0
        count_decl_instance_method = 1 if ("::" in signature and "static" not in signature) else 0

        # Count input and output operations.
        count_input = len(re.findall(r'\b(scanf|cin)\b', code))
        count_output = len(re.findall(r'\b(printf|cout)\b', code))
        count_semicolon = code.count(';')

        # Code smells (very naive – for example, too long function or too many parameters)
        code_smells_count = 0
        if loc_total > 100:
            code_smells_count += 1
        if parameter_count > 5:
            code_smells_count += 1

        # Inheritance depth (if the signature shows a method from a class, assume at least depth 1)
        inheritance_depth = 1 if "::" in signature else 0

        # Coupling between objects: approximated by fan_out_direct.
        coupling_between_objects = fan_out_direct
        # Lack of cohesion: not computed at function level.
        lack_of_cohesion_of_methods = 0
        # Weighted methods per class: use cyclomatic complexity as proxy.
        weighted_methods_per_class = cyclomatic_complexity

        # Maintainability index (MI) using a common formula.
        try:
            # Use original formula, which is sensitive to zero/negative values
            mi = 171 - 5.2 * math.log(halstead["HALSTEAD_VOLUME"]) - 0.23 * cyclomatic_complexity - 16.2 * math.log(loc_executable)
        except ValueError:
            mi = 0
            
        maintainability_index = mi
        refactorability_index = maintainability_index  # using MI as a proxy

        # Cognitive Complexity: approximated by cyclomatic complexity.
        cognitive_complexity = cyclomatic_complexity

        # Code duplication ratio: fraction of duplicate nonblank lines.
        nonblank_lines = [line.strip() for line in lines if line.strip() != ""]
        if nonblank_lines:
            duplicate_count = len(nonblank_lines) - len(set(nonblank_lines))
            code_duplication_ratio = duplicate_count / len(nonblank_lines)
        else:
            code_duplication_ratio = 0

        # Static code warnings: as a placeholder, we assume 0.
        static_code_warnings = 0

        # Compile all metrics in a dictionary.
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