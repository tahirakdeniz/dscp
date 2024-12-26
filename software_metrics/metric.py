import lizard
import re
import re
from collections import defaultdict
import csv
import os
import glob
import time
import sys

def calculate_complexity_metrics(file_path):
    # Analyze the file using lizard
    lizard_analysis = lizard.analyze_file(file_path)
    
    # Extract cyclomatic complexity
    cyclomatic_complexity = lizard_analysis.function_list[0].cyclomatic_complexity if lizard_analysis.function_list else 0

    # Extract Halstead metrics
    if lizard_analysis.function_list:
        function_info = lizard_analysis.function_list[0]
        vocabulary = function_info.token_count
        length = function_info.length
        volume = length * (vocabulary * 2).bit_length()
        difficulty = (function_info.parameter_count / 2) * (length / vocabulary)
        effort = difficulty * volume
    else:
        vocabulary = length = volume = difficulty = effort = 0
    
    return {
        'cyclomatic_complexity': cyclomatic_complexity,
        'vocabulary': vocabulary,
        'length': length,
        'volume': volume,
        'difficulty': difficulty,
        'effort': effort
    }

def calculate_size_metrics(file_path):
    total_loc = 0
    code_loc = 0
    comment_loc = 0
    in_block_comment = False

    with open(file_path, 'r') as file:
        for line in file:
            total_loc += 1
            stripped_line = line.strip()
            
            if stripped_line.startswith('/*'):
                in_block_comment = True
                comment_loc += 1
            elif stripped_line.endswith('*/'):
                in_block_comment = False
                comment_loc += 1
            elif in_block_comment or stripped_line.startswith('//'):
                comment_loc += 1
            elif stripped_line:
                code_loc += 1

    comment_density = comment_loc / total_loc if total_loc else 0

    return {
        'total_loc': total_loc,
        'code_loc': code_loc,
        'comment_loc': comment_loc,
        'comment_density': comment_density
    }

def calculate_structural_metrics(file_path):
    lizard_analysis = lizard.analyze_file(file_path)
    
    function_lengths = [func.length for func in lizard_analysis.function_list]
    num_parameters = [func.parameter_count for func in lizard_analysis.function_list]
    
    # Calculate max depth of nesting manually
    def calculate_max_nesting_depth(code):
        max_depth = 0
        current_depth = 0
        for char in code:
            if char == '{':
                current_depth += 1
                if current_depth > max_depth:
                    max_depth = current_depth
            elif char == '}':
                current_depth -= 1
        return max_depth

    with open(file_path, 'r') as file:
        code = file.read()
    
    max_depth_of_nesting = calculate_max_nesting_depth(code)
    
    avg_function_length = sum(function_lengths) / len(function_lengths) if function_lengths else 0
    avg_num_parameters = sum(num_parameters) / len(num_parameters) if num_parameters else 0

    num_functions = len(function_lengths)
    max_function_length = max(function_lengths) if function_lengths else 0
    min_function_length = min(function_lengths) if function_lengths else 0
    max_num_parameters = max(num_parameters) if num_parameters else 0

    return {
        'max_depth_of_nesting': max_depth_of_nesting,
        'avg_function_length': avg_function_length,
        'avg_num_parameters': avg_num_parameters,
        'num_functions': num_functions,
        'max_function_length': max_function_length,
        'min_function_length': min_function_length,
        'max_num_parameters': max_num_parameters
    }

def calculate_coupling_cohesion_metrics(file_path):
    """
    Calculate coupling and cohesion metrics for a C/C++ source file.
    
    Args:
        file_path: Path to the source code file
        
    Returns:
        Dictionary containing:
        - num_global_vars: Number of global variables
        - num_external_calls: Number of calls to functions not defined in the file
        - num_internal_calls: Number of calls to functions defined in the file
    """
    num_global_vars = 0
    num_external_calls = 0
    num_internal_calls = 0
    functions = set()
    
    # Common variable types in C/C++
    var_types = r'^(int|float|double|char|long|short|unsigned|bool)\s+'
    
    # Function declaration pattern
    func_decl_pattern = re.compile(r'^[\w\s]+\s+(\w+)\s*\([^)]*\)\s*{')
    
    # Function call pattern
    func_call_pattern = re.compile(r'(\w+)\s*\([^)]*\)')
    
    # Read entire file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split into lines while preserving line breaks
    lines = content.split('\n')
    
    # First pass: collect function declarations and global variables
    in_function = False
    for line in lines:
        stripped_line = line.strip()
        
        # Skip empty lines and comments
        if not stripped_line or stripped_line.startswith('//') or stripped_line.startswith('/*'):
            continue
            
        # Check if we're entering a function
        if not in_function and '{' in line:
            func_match = func_decl_pattern.search(line)
            if func_match:
                functions.add(func_match.group(1))
                in_function = True
                continue
        
        # Check if we're exiting a function
        if in_function and '}' in line:
            in_function = False
            continue
            
        # Count global variables (only if not in function)
        if not in_function and re.match(var_types, stripped_line):
            # Split multiple declarations
            declarations = stripped_line.split(',')
            num_global_vars += len(declarations)
    
    # Second pass: count function calls
    for line in lines:
        stripped_line = line.strip()
        
        # Skip comments and preprocessor directives
        if not stripped_line or stripped_line.startswith('//') or stripped_line.startswith('#'):
            continue
            
        # Find all function calls in the line
        for match in func_call_pattern.finditer(stripped_line):
            func_name = match.group(1)
            
            # Skip control structures
            if func_name in {'if', 'for', 'while', 'switch'}:
                continue
                
            if func_name in functions:
                num_internal_calls += 1
            else:
                num_external_calls += 1
    
    return {
        'num_global_vars': num_global_vars,
        'num_external_calls': num_external_calls,
        'num_internal_calls': num_internal_calls
    }

def calculate_performance_metrics(file_path):
    loop_complexity = {
        'num_loops': 0,
        'max_nesting_depth': 0
    }
    memory_usage = {
        'static_allocations': 0,
        'dynamic_allocations': 0
    }

    loop_patterns = [re.compile(r'\bfor\s*\('), re.compile(r'\bwhile\s*\(')]
    static_alloc_pattern = re.compile(r'\b(int|float|double|char|long|short|unsigned|bool)\s+\w+\s*(\[.*\])')
    dynamic_alloc_pattern = re.compile(r'\bmalloc\s*\(')
    free_pattern = re.compile(r'\bfree\s*\(')

    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_loop_depth = 0
    for line in lines:
        stripped_line = line.strip()

        # Check for loop patterns
        for pattern in loop_patterns:
            if pattern.search(stripped_line):
                loop_complexity['num_loops'] += 1
                current_loop_depth += 1
                if current_loop_depth > loop_complexity['max_nesting_depth']:
                    loop_complexity['max_nesting_depth'] = current_loop_depth

        # Check for end of loop
        if '}' in stripped_line:
            current_loop_depth = max(0, current_loop_depth - 1)

        # Check for static memory allocations
        if static_alloc_pattern.search(stripped_line):
            memory_usage['static_allocations'] += 1

        # Check for dynamic memory allocations
        if dynamic_alloc_pattern.search(stripped_line):
            memory_usage['dynamic_allocations'] += 1

        # Check for free calls
        if free_pattern.search(stripped_line):
            memory_usage['dynamic_allocations'] -= 1

    return {
        'num_loops': loop_complexity['num_loops'],
        'max_loop_nesting_depth': loop_complexity['max_nesting_depth'],
        'static_memory_allocations': memory_usage['static_allocations'],
        'dynamic_memory_allocations': memory_usage['dynamic_allocations']
    }

def calculate_custom_metrics(file_path):
    """
    Analyzes a C file for security checks and API usage patterns.
    
    Args:
        file_path: Path to the C source file
        
    Returns:
        Dictionary containing:
        - security_metrics: Dict with counts of different security checks
        - api_usage: Dict with counts of API/library function calls
    """
    # Initialize metrics
    security_metrics = {
        'input_validations': 0,    # if(...) checks on input parameters
        'bounds_checks': 0,        # array bounds checks
        'null_checks': 0,          # NULL pointer checks
        'error_handling': 0,       # try/catch, if(error), etc.
        'memory_checks': 0,        # malloc checks, size checks
        'sanitization_checks': 0   # input sanitization
    }
    
    api_usage = defaultdict(int)
    
    # Common security-related functions and patterns
    security_functions = {
        'malloc': 'memory_checks',
        'calloc': 'memory_checks',
        'free': 'memory_checks',
        'strcpy_s': 'bounds_checks',
        'strncpy': 'bounds_checks',
        'memcpy': 'bounds_checks',
        'scanf': 'input_validations',
        'fscanf': 'input_validations',
        'gets': 'input_validations',
        'fgets': 'input_validations',
    }
    
    # Common APIs and libraries to track
    common_apis = {
        # Standard C library
        'printf', 'sprintf', 'fprintf',
        'malloc', 'free', 'calloc', 'realloc',
        'fopen', 'fclose', 'fread', 'fwrite',
        'strcpy', 'strncpy', 'strcat', 'strncat',
        # Network functions
        'socket', 'connect', 'bind', 'listen',
        # Cryptographic functions
        'encrypt', 'decrypt', 'hash',
        # Thread functions
        'pthread_create', 'pthread_join'
    }
    
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            lines = content.split('\n')
            
        in_comment = False
        
        for i, line in enumerate(lines):
            # Skip comments
            if '/*' in line:
                in_comment = True
            if '*/' in line:
                in_comment = False
                continue
            if in_comment or line.strip().startswith('//'):
                continue
                
            # Check for NULL pointer validation
            if 'NULL' in line or 'null' in line:
                if 'if' in line.lower() and ('==' in line or '!=' in line):
                    security_metrics['null_checks'] += 1
            
            # Check for bounds checking
            if 'if' in line.lower() and any(x in line.lower() for x in ['length', 'len', 'size', '<', '>', '<=', '>=']):
                if any(x in line.lower() for x in ['array', 'str', 'buf', 'pointer']):
                    security_metrics['bounds_checks'] += 1
            
            # Check for error handling
            if any(x in line.lower() for x in ['error', 'exception', 'errno', 'return -1', 'exit']):
                if 'if' in line.lower() or 'catch' in line.lower():
                    security_metrics['error_handling'] += 1
            
            # Check for input validation
            if 'if' in line.lower():
                if any(x in line.lower() for x in ['input', 'argv', 'scanf', 'gets', 'fgets', 'read']):
                    security_metrics['input_validations'] += 1
            
            # Check for sanitization
            if any(x in line.lower() for x in ['sanitize', 'escape', 'validate', 'clean']):
                security_metrics['sanitization_checks'] += 1
            
            # Track API usage
            words = re.findall(r'\w+', line)
            for word in words:
                if word in common_apis:
                    api_usage[word] += 1
                    
            # Check security functions
            for func, metric_type in security_functions.items():
                if func in line and '(' in line:
                    security_metrics[metric_type] += 1
    
    except Exception as e:
        return {
            'error': f'Failed to analyze file: {str(e)}',
            'security_metrics': security_metrics,
            'api_usage': dict(api_usage)
        }
    
    # Group API usage by category
    categorized_api_usage = {
        'memory_management': sum(api_usage[api] for api in ['malloc', 'free', 'calloc', 'realloc']),
        'io_operations': sum(api_usage[api] for api in ['printf', 'scanf', 'fopen', 'fclose', 'fread', 'fwrite']),
        'string_operations': sum(api_usage[api] for api in ['strcpy', 'strncpy', 'strcat', 'strncat']),
        'network_operations': sum(api_usage[api] for api in ['socket', 'connect', 'bind', 'listen']),
        'crypto_operations': sum(api_usage[api] for api in ['encrypt', 'decrypt', 'hash']),
        'threading': sum(api_usage[api] for api in ['pthread_create', 'pthread_join']),
        'detailed_usage': dict(api_usage)
    }
    
    return {
        **security_metrics,
        **categorized_api_usage
    }

def calculate_metrics(file_path):
    
    size_metrics = calculate_size_metrics(file_path)
    complexity_metrics = calculate_complexity_metrics(file_path)
    structural_metrics = calculate_structural_metrics(file_path)
    coupling_cohesion_metrics = calculate_coupling_cohesion_metrics(file_path)
    perfomance_metrics = calculate_performance_metrics(file_path)
    custom_metrics = calculate_custom_metrics(file_path)
    
    return {
        **complexity_metrics,
        **size_metrics,
        **structural_metrics,
        **coupling_cohesion_metrics,
        **perfomance_metrics,
        **custom_metrics
    }
    
def dict_to_csv(file_name, metrics_dict, output_csv="results/metrics.csv"):
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    file_exists = os.path.isfile(output_csv)
    
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header only if the file does not exist
        if not file_exists:
            headers = ['file_name'] + list(metrics_dict.keys())
            writer.writerow(headers)
        
        # Write the data
        row = [file_name] + list(metrics_dict.values())
        writer.writerow(row)

def process_files(file_paths):
    start_time = time.time()
    for i, file_path in enumerate(file_paths):
        metrics = calculate_metrics(file_path)
        dict_to_csv(file_path, metrics)
        if (i + 1) % 1000 == 0:
            end_time = time.time()
            print(f"Processed {i + 1} files. Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    
    print("Starting analysis...")
    
    if len(sys.argv) != 2:
        print("Usage: python main.py <directory_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    file_path_list = glob.glob(f'{file_path}/*.c')
    
    process_files(file_path_list)