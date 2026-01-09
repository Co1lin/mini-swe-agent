"""
Script to evaluate the localization accuracy.

This script evaluates localization accuracy at multiple levels:
1. File-level: What fraction of files are correctly localized?
2. Function-level: What fraction of functions are correctly localized?
3. Fine-grained line level: What fraction of specific edit lines are correctly localized?

For each level, it calculates:
- Precision: What fraction of predicted items are correct?
- Recall: What fraction of ground-truth items are found?
- Contains GT: What fraction of problems have all ground-truth items in the predicted set?
"""

import json
import argparse
import re
import ast
from pathlib import Path
from typing import Dict, Set, Optional, Union
from collections import defaultdict
import tempfile

import subprocess
import shutil

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not found. Please install it with: pip install datasets")
    exit(1)


# Global cache for repo paths and their current commit state
_repo_cache: Dict[str, Path] = {}  # repo -> repo_path
_repo_current_commit: Dict[str, str] = {}  # repo -> current_commit
REPOS_DIR = None


def setup_repo(repo: str, base_commit: str) -> Optional[Path]:
    """
    Clone a repository (if needed) and checkout to the specified commit.

    The repo is cloned once and reused. We track the current commit state
    and only checkout when the required commit differs from the current one.

    Args:
        repo: Repository in format "owner/name" (e.g., "django/django")
        base_commit: The commit hash to checkout to

    Returns:
        Path to the repository root, or None if setup failed
    """
    global _repo_cache, _repo_current_commit

    # REPOS_DIR.mkdir(parents=True, exist_ok=False)

    # Repository directory name
    repo_name = repo.replace("/", "__")
    repo_path = REPOS_DIR / repo_name

    try:
        if repo not in _repo_cache:
            if repo_path.exists():
                # remove the repository
                print(f"Repository {repo_path} already exists, removing it...")
                shutil.rmtree(repo_path)
            # Clone the repository
            clone_url = f"https://github.com/{repo}.git"
            print(f"Cloning repository {clone_url} to {repo_path}...")
            subprocess.run(
                ["git", "clone", "--quiet", clone_url, str(repo_path)],
                check=True,
                capture_output=True,
                timeout=300
            )
            _repo_cache[repo] = repo_path

        # Check if we need to checkout to a different commit
        current_commit = _repo_current_commit.get(repo)
        if current_commit != base_commit:
            # Checkout to the specific commit
            print(f"Checking out to commit {base_commit}...")
            subprocess.run(
                ["git", "checkout", "--quiet", "--force", base_commit],
                cwd=repo_path,
                check=True,
                capture_output=True,
                timeout=60
            )
            _repo_current_commit[repo] = base_commit

        return repo_path

    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to setup repo {repo}@{base_commit}: {e}")
        return None
    except subprocess.TimeoutExpired:
        print(f"Warning: Timeout setting up repo {repo}@{base_commit}")
        return None


def get_functions_at_lines(source_code: str, lines: Set[int]) -> Set[str]:
    """
    Parse Python source code and find which functions/classes contain the given lines.

    Args:
        source_code: The Python source code to parse
        lines: Set of line numbers to look up

    Returns:
        Set of function/class identifiers (e.g., "ClassName.method_name", "function_name")
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return {"<parse_error>"}

    # Build a mapping of line ranges to function/class names
    # We need to track the hierarchy: module -> class -> method
    functions = set()

    class FunctionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.class_stack = []  # Stack of class names we're inside

        def visit_ClassDef(self, node):
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node):
            # Check if any of our target lines fall within this function
            func_start = node.lineno
            func_end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else func_start

            for line in lines:
                if func_start <= line <= func_end:
                    if self.class_stack:
                        func_id = f"{'.'.join(self.class_stack)}.{node.name}"
                    else:
                        func_id = node.name
                    functions.add(func_id)
                    break

            # Visit nested functions/classes
            old_class_stack = self.class_stack.copy()
            self.class_stack = []  # Reset for nested functions (they're not methods)
            self.generic_visit(node)
            self.class_stack = old_class_stack

        def visit_AsyncFunctionDef(self, node):
            # Same logic as FunctionDef
            self.visit_FunctionDef(node)

    visitor = FunctionVisitor()
    visitor.visit(tree)

    # If no functions found, check if lines are at class body level or module level
    if not functions:
        class ScopeVisitor(ast.NodeVisitor):
            def __init__(self):
                self.class_stack = []

            def visit_ClassDef(self, node):
                class_start = node.lineno
                class_end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else class_start

                for line in lines:
                    if class_start <= line <= class_end:
                        # Line is in class, but we already checked functions
                        # So it must be at class body level
                        self.class_stack.append(node.name)
                        self.generic_visit(node)
                        # Check if line is directly in this class (not in a method)
                        # by checking if we found it in a nested function
                        if not functions:
                            functions.add('.'.join(self.class_stack))
                        self.class_stack.pop()
                        return

                self.generic_visit(node)

        scope_visitor = ScopeVisitor()
        scope_visitor.visit(tree)

    # If still no functions found, it's at module level
    if not functions:
        functions.add("<module>")

    return functions


def extract_file_content_from_patch(patch: str, file_path: str) -> Optional[str]:

    """
    Extract the full content of a newly created file from a patch.

    Args:
        patch: The git diff patch
        file_path: The path of the file to extract

    Returns:
        The file content, or None if not a new file
    """
    lines = patch.split('\n')
    in_target_file = False
    is_new_file = False
    content_lines = []

    for line in lines:
        if line.startswith('diff --git'):
            # Check if this is our target file
            match = re.search(r'diff --git a/(.+?) b/(.+?)$', line)
            if match and match.group(2) == file_path:
                in_target_file = True
                is_new_file = False
                content_lines = []
            else:
                in_target_file = False
        elif in_target_file:
            if line.startswith('new file mode'):
                is_new_file = True
            elif line.startswith('--- /dev/null'):
                is_new_file = True
            elif line.startswith('+') and not line.startswith('+++'):
                if is_new_file:
                    content_lines.append(line[1:])  # Remove the '+' prefix

    if is_new_file and content_lines:
        return '\n'.join(content_lines)
    return None


def parse_patch_touched_lines(patch: str, use_original_coords: bool = False) -> Dict[str, Set[int]]:
    """
    Parse a git diff patch to extract "touched lines".

    Touched lines are the union of:
    - Deleted line numbers (in original file coordinates) - only for existing files
    - Added line numbers (coordinates depend on use_original_coords parameter)

    Args:
        patch: Git diff patch string
        use_original_coords: If True, use original file coordinates for added lines
            (insertion point). This is useful for function-level analysis where we
            need to look up line numbers in the original file. If False (default),
            use new file coordinates for added lines.

    Returns:
        Dictionary mapping file paths to sets of touched line numbers
    """
    file_lines = defaultdict(set)
    lines = patch.split('\n')
    current_file = None
    current_original_line = 0
    current_new_line = 0
    is_new_file = False

    for line in lines:
        if line.startswith('diff --git'):
            match = re.search(r'diff --git a/(.+?) b/(.+?)$', line)
            if match:
                current_file = match.group(2)
                is_new_file = False
        elif line.startswith('new file mode') or line.startswith('--- /dev/null'):
            is_new_file = True
        elif line.startswith('+++') and not line.startswith('+++ /dev/null'):
            match = re.search(r'\+\+\+ (?:b/)?(.+)$', line)
            if match:
                current_file = match.group(1)
        elif line.startswith('@@'):
            # Extract line numbers from hunk header
            # Format: @@ -OLD_START,OLD_COUNT +NEW_START,NEW_COUNT @@
            match = re.search(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
            if match:
                current_original_line = int(match.group(1))
                current_new_line = int(match.group(2))
        elif current_file and line.startswith('-') and not line.startswith('---'):
            # Deleted line: use original file line number (skip for new files)
            # Skip empty lines (blank or whitespace only)
            line_content = line[1:]  # Remove the '-' prefix
            if not is_new_file and line_content.strip():
                file_lines[current_file].add(current_original_line)
            current_original_line += 1
        elif current_file and line.startswith('+') and not line.startswith('+++'):
            # Added line: use original coords (insertion point) or new coords
            # Skip empty lines (blank or whitespace only)
            line_content = line[1:]  # Remove the '+' prefix
            if line_content.strip():
                if use_original_coords:
                    # Use insertion point in original file for function-level analysis
                    file_lines[current_file].add(current_original_line)
                else:
                    # Use new file line number for line-level analysis
                    file_lines[current_file].add(current_new_line)
            current_new_line += 1
        elif current_file and not line.startswith('+') and not line.startswith('-') and not line.startswith('\\'):
            # Context line: increment both counters
            current_original_line += 1
            current_new_line += 1

    return dict(file_lines)


def apply_patch_to_content(original_content: str, patch: str, file_path: str) -> Optional[str]:
    """
    Apply a patch to file content and return the modified content.

    Args:
        original_content: The original file content
        patch: The full git diff patch
        file_path: The path of the file to extract changes for

    Returns:
        The modified file content, or None if patch application fails
    """
    lines = original_content.split('\n')
    patch_lines = patch.split('\n')

    in_target_file = False
    hunks = []
    current_hunk = None

    # Parse the patch to extract hunks for this file
    for line in patch_lines:
        if line.startswith('diff --git'):
            match = re.search(r'diff --git a/(.+?) b/(.+?)$', line)
            if match and match.group(2) == file_path:
                in_target_file = True
            else:
                in_target_file = False
        elif in_target_file and line.startswith('@@'):
            match = re.search(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if match:
                if current_hunk:
                    hunks.append(current_hunk)
                current_hunk = {
                    'old_start': int(match.group(1)),
                    'old_count': int(match.group(2)) if match.group(2) else 1,
                    'new_start': int(match.group(3)),
                    'new_count': int(match.group(4)) if match.group(4) else 1,
                    'lines': []
                }

        elif in_target_file and current_hunk is not None:
            if line.startswith('+') and not line.startswith('+++'):
                current_hunk['lines'].append(('+', line[1:]))
            elif line.startswith('-') and not line.startswith('---'):
                current_hunk['lines'].append(('-', line[1:]))
            elif not line.startswith('\\'):
                current_hunk['lines'].append((' ', line[1:] if line.startswith(' ') else line))

    if current_hunk:
        hunks.append(current_hunk)

    if not hunks:
        return None

    # Apply hunks in reverse order to preserve line numbers
    result_lines = lines[:]
    for hunk in reversed(hunks):
        old_start = hunk['old_start'] - 1  # Convert to 0-indexed
        old_end = old_start + hunk['old_count']

        new_lines = []
        for op, content in hunk['lines']:
            if op in ('+', ' '):
                new_lines.append(content)



        result_lines = result_lines[:old_start] + new_lines + result_lines[old_end:]

    return '\n'.join(result_lines)


def parse_patch_deleted_lines(patch: str) -> Dict[str, Set[int]]:
    """Parse a patch to get only the deleted lines (original file coordinates).

    Empty lines (blank or whitespace only) are ignored.
    """
    file_lines = defaultdict(set)
    lines = patch.split('\n')
    current_file = None
    current_original_line = 0
    is_new_file = False

    for line in lines:
        if line.startswith('diff --git'):
            match = re.search(r'diff --git a/(.+?) b/(.+?)$', line)
            if match:
                current_file = match.group(2)
                is_new_file = False
        elif line.startswith('new file mode') or line.startswith('--- /dev/null'):
            is_new_file = True
        elif line.startswith('@@'):
            match = re.search(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
            if match:
                current_original_line = int(match.group(1))
        elif current_file and line.startswith('-') and not line.startswith('---'):
            # Skip empty lines (blank or whitespace only)
            line_content = line[1:]  # Remove the '-' prefix
            if not is_new_file and line_content.strip():
                file_lines[current_file].add(current_original_line)
            current_original_line += 1
        elif current_file and line.startswith('+') and not line.startswith('+++'):
            pass  # Skip added lines
        elif current_file and not line.startswith('+') and not line.startswith('-') and not line.startswith('\\'):
            current_original_line += 1

    return dict(file_lines)


def parse_patch_added_lines(patch: str) -> Dict[str, Set[int]]:
    """Parse a patch to get only the added lines (new file coordinates).

    Empty lines (blank or whitespace only) are ignored.
    """
    file_lines = defaultdict(set)
    lines = patch.split('\n')
    current_file = None
    current_new_line = 0

    for line in lines:
        if line.startswith('diff --git'):
            match = re.search(r'diff --git a/(.+?) b/(.+?)$', line)
            if match:
                current_file = match.group(2)
        elif line.startswith('@@'):
            match = re.search(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
            if match:
                current_new_line = int(match.group(2))
        elif current_file and line.startswith('-') and not line.startswith('---'):
            pass  # Skip deleted lines, don't increment new line counter
        elif current_file and line.startswith('+') and not line.startswith('+++'):
            # Skip empty lines (blank or whitespace only)
            line_content = line[1:]  # Remove the '+' prefix
            if line_content.strip():
                file_lines[current_file].add(current_new_line)
            current_new_line += 1
        elif current_file and not line.startswith('+') and not line.startswith('-') and not line.startswith('\\'):
            current_new_line += 1

    return dict(file_lines)


def parse_patch_functions_ast(patch: str, repo_path: Optional[Path] = None) -> Dict[str, Set[str]]:
    """
    Parse a patch to extract functions that contain modifications using AST.

    For deleted lines: looks up in the original file
    For added lines: looks up in the modified file (after applying patch)

    Args:
        patch: The git diff patch
        repo_path: Path to the repository (for reading existing files)

    Returns:
        Dictionary mapping file paths to sets of function identifiers
    """
    file_functions = defaultdict(set)

    # Get deleted and added lines separately
    deleted_lines_by_file = parse_patch_deleted_lines(patch)
    added_lines_by_file = parse_patch_added_lines(patch)

    # Get all affected files
    all_files = set(deleted_lines_by_file.keys()) | set(added_lines_by_file.keys())


    for file_path in all_files:
        if not file_path.endswith('.py'):
            # Skip non-Python files, just record <non-python>
            file_functions[file_path].add("<non-python>")
            continue

        deleted_lines = deleted_lines_by_file.get(file_path, set())
        added_lines = added_lines_by_file.get(file_path, set())

        # Check if it's a new file
        new_file_content = extract_file_content_from_patch(patch, file_path)

        if new_file_content:
            # New file: use the new file content for added lines
            if added_lines:
                functions = get_functions_at_lines(new_file_content, added_lines)
                file_functions[file_path].update(functions)
        elif repo_path:
            # Existing file: handle deleted and added lines separately
            full_path = repo_path / file_path
            if full_path.exists():
                try:
                    original_content = full_path.read_text()

                    # For deleted lines: look up in original file
                    if deleted_lines:
                        functions = get_functions_at_lines(original_content, deleted_lines)
                        file_functions[file_path].update(functions)

                    # For added lines: apply patch and look up in modified file
                    if added_lines:
                        modified_content = apply_patch_to_content(original_content, patch, file_path)
                        if modified_content:
                            functions = get_functions_at_lines(modified_content, added_lines)
                            file_functions[file_path].update(functions)
                        else:
                            # Fallback: use original file with insertion point approximation
                            file_functions[file_path].add("<patch_apply_failed>")
                except Exception as e:
                    file_functions[file_path].add("<unknown>")
            else:
                file_functions[file_path].add("<unknown>")
        else:
            file_functions[file_path].add("<unknown>")

    return dict(file_functions)


def parse_patch_files(patch: str) -> Set[str]:
    """
    Parse a git diff patch to extract the files that are modified.

    Args:
        patch: Git diff patch string

    Returns:
        Set of file paths that are modified in the patch
    """
    files = set()

    # Look for lines starting with "diff --git" or "+++" or "---"
    # Example patterns:
    # diff --git a/file.py b/file.py
    # --- a/file.py
    # +++ b/file.py

    lines = patch.split('\n')
    for line in lines:
        # Match diff --git a/path b/path
        if line.startswith('diff --git'):
            # Extract file path from "diff --git a/path b/path"
            match = re.search(r'diff --git a/(.+?) b/(.+?)$', line)
            if match:
                file_path = match.group(2)  # Use the 'b/' path (after modification)
                files.add(file_path)

        # Alternative: extract from +++ lines (files being added/modified)
        elif line.startswith('+++') and not line.startswith('+++ /dev/null'):
            # Extract path from "+++ b/path" or "+++ path"
            match = re.search(r'\+\+\+ (?:b/)?(.+)$', line)
            if match:
                file_path = match.group(1)
                files.add(file_path)

    return files


def load_predictions(pred_file: str, evaluation_level: str = "file",
                    dataset_name: str = "princeton-nlp/SWE-bench_Verified") -> Dict[str, Union[Set[str], Dict[str, Set]]]:
    """
    Load predictions from a JSON file containing generated patches.

    Extracts locations from the model_patch field by parsing the generated diffs.

    Args:
        pred_file: Path to the predictions JSON file
        evaluation_level: Level of evaluation ("file", "function", "line")
        dataset_name: Dataset name to get repo/commit info for AST parsing

    Returns:
        Dictionary mapping instance_id to predicted items (extracted from patches)
    """
    predictions = {}

    with open(pred_file, 'r') as f:
        data_list = json.load(f).values()

    # For function-level evaluation, we need repo info from the dataset
    instance_info = {}
    if evaluation_level == "function":
        print("Loading dataset for repo info...")
        dataset = load_dataset(dataset_name, split="test")
        for item in dataset:
            instance_info[item['instance_id']] = {
                'repo': item['repo'],
                'base_commit': item['base_commit']
            }

    for data in data_list:
        instance_id = data['instance_id']
        patch = data.get('model_patch', '')

        if evaluation_level == "file":
            found_items = parse_patch_files(patch) if patch else set()
        elif evaluation_level == "function":
            if patch and instance_id in instance_info:
                info = instance_info[instance_id]
                repo_path = setup_repo(info['repo'], info['base_commit'])
                found_items = parse_patch_functions_ast(patch, repo_path)
            else:
                found_items = {} if not patch else parse_patch_functions_ast(patch, None)
        elif evaluation_level == "line":
            found_items = parse_patch_touched_lines(patch) if patch else {}
        else:
            raise ValueError(f"Unknown evaluation level: {evaluation_level}")

        predictions[instance_id] = found_items

    return predictions


def load_ground_truth(
    predictions: Dict[str, Union[Set[str], Dict[str, Set]]],
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
                     evaluation_level: str = "file") -> Dict[str, Union[Set[str], Dict[str, Set]]]:
    """
    Load ground truth from SWE-bench dataset.

    Args:
        dataset_name: Name of the dataset to load
        evaluation_level: Level of evaluation ("file", "function", "line")

    Returns:
        Dictionary mapping instance_id to ground-truth items
    """
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split="test")

    ground_truth = {}

    # only load the instances that are in the predictions
    dataset = dataset.filter(lambda x: x['instance_id'] in predictions.keys())


    for item in dataset:
        instance_id = item['instance_id']
        patch = item['patch']

        if evaluation_level == "file":
            # Parse the patch to extract modified files
            gt_items = parse_patch_files(patch)
        elif evaluation_level == "function":
            # Use AST-based parsing for accurate function detection
            repo_path = setup_repo(item['repo'], item['base_commit'])
            gt_items = parse_patch_functions_ast(patch, repo_path)
        elif evaluation_level == "line":
            # Parse the patch to extract touched lines
            gt_items = parse_patch_touched_lines(patch)
        else:
            raise ValueError(f"Unknown evaluation level: {evaluation_level}")

        ground_truth[instance_id] = gt_items

    return ground_truth


def calculate_metrics(predictions: Dict[str, Union[Set[str], Dict[str, Set]]],
                     ground_truth: Dict[str, Union[Set[str], Dict[str, Set]]],
                     evaluation_level: str = "file") -> Dict[str, float]:
    """
    Calculate precision, recall, and Contains GT metrics.

    Args:
        predictions: Dictionary mapping instance_id to predicted items (extracted from patches)
        ground_truth: Dictionary mapping instance_id to ground-truth items
        evaluation_level: Level of evaluation ("file", "function", "line")

    Returns:
        Dictionary with calculated metrics
    """
    total_problems = 0
    total_intersection = 0  # TP
    total_pred_files = 0  # TP + FP
    total_gt_files = 0  # TP + FN
    contains_gt_count = 0

    # For detailed analysis
    precision_scores = []
    recall_scores = []

    # Find common instances
    common_instances = set(predictions.keys()) & set(ground_truth.keys())

    if not common_instances:
        print("Warning: No common instances found between predictions and ground truth!")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'contains_gt': 0.0,
            'total_problems': 0
        }

    for instance_id in common_instances:
        pred_items = predictions[instance_id]
        gt_items = ground_truth[instance_id]

        if evaluation_level == "file":
            # File-level evaluation
            pred_files = set(pred_items) if pred_items else set()
            gt_files = gt_items

            # Skip instances with no ground truth files
            if not gt_files:
                raise ValueError(f"No ground truth files found for instance {instance_id}")

            total_problems += 1

            # Calculate precision and recall
            intersection = pred_files & gt_files
            precision = len(intersection) / len(pred_files) if pred_files else 0.0
            recall = len(intersection) / len(gt_files) if gt_files else 0.0

            total_intersection += len(intersection)
            total_pred_files += len(pred_files)
            total_gt_files += len(gt_files)

            precision_scores.append(precision)
            recall_scores.append(recall)

            # Contains GT: check if ALL ground-truth files are in predicted set
            if gt_files.issubset(pred_files):
                contains_gt_count += 1

        elif evaluation_level in ["function", "line"]:
            # Function or line-level evaluation
            # pred_items is already parsed (dict with sets) from load_predictions
            pred_elements = pred_items if pred_items else {}
            gt_elements = gt_items if gt_items else {}

            total_problems += 1

            # Flatten predictions and ground truth for comparison
            pred_flat = set()
            gt_flat = set()

            for file_path, elements in pred_elements.items():
                for element in elements:
                    pred_flat.add(f"{file_path}::{element}")

            for file_path, elements in gt_elements.items():
                for element in elements:
                    gt_flat.add(f"{file_path}::{element}")

            # Calculate precision and recall
            intersection = pred_flat & gt_flat
            precision = len(intersection) / len(pred_flat) if pred_flat else 0.0
            recall = len(intersection) / len(gt_flat) if gt_flat else 0.0

            total_intersection += len(intersection)
            total_pred_files += len(pred_flat)
            total_gt_files += len(gt_flat)

            precision_scores.append(precision)
            recall_scores.append(recall)

            # Contains GT: check if ALL ground-truth elements are in predicted set
            if gt_flat.issubset(pred_flat):
                contains_gt_count += 1

    # Calculate averages
    avg_precision = total_intersection / total_pred_files if total_pred_files > 0 else 0.0
    avg_recall = total_intersection / total_gt_files if total_gt_files > 0 else 0.0
    contains_gt_percentage = (contains_gt_count / total_problems * 100) if total_problems > 0 else 0.0

    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'contains_gt': contains_gt_percentage,
        'total_problems': total_problems,
        'precision_scores': precision_scores,
        'recall_scores': recall_scores
    }


def print_detailed_analysis(predictions: Dict[str, Union[Set[str], Dict[str, Set]]],
                           ground_truth: Dict[str, Union[Set[str], Dict[str, Set]]],
                           evaluation_level: str = "file",
                           limit: int = 5):
    """
    Print detailed analysis for a few examples.

    Args:
        predictions: Dictionary mapping instance_id to predicted items
        ground_truth: Dictionary mapping instance_id to ground-truth items
        evaluation_level: Level of evaluation ("file", "function", "line")
        limit: Number of examples to show
    """
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS - {evaluation_level.upper()} LEVEL (First {limit} examples)")
    print('='*60)

    common_instances = list(set(predictions.keys()) & set(ground_truth.keys()))

    for i, instance_id in enumerate(common_instances[:limit]):
        pred_items = predictions[instance_id]
        gt_items = ground_truth[instance_id]

        print(f"\nExample {i+1}: {instance_id}")

        if evaluation_level == "file":
            pred_files = pred_items if pred_items else set()
            gt_files = gt_items

            print(f"Ground Truth Files ({len(gt_files)}):")
            for f in sorted(gt_files):
                print(f"  - {f}")

            print(f"Predicted Files ({len(pred_files)}):")
            for f in sorted(pred_files):
                marker = "✓" if f in gt_files else "✗"
                print(f"  {marker} {f}")

            intersection = pred_files & gt_files
            precision = len(intersection) / len(pred_files) if pred_files else 0.0
            recall = len(intersection) / len(gt_files) if gt_files else 0.0
            contains_gt = gt_files.issubset(pred_files)

        elif evaluation_level == "function":
            pred_funcs = pred_items if pred_items else {}
            gt_funcs = gt_items if gt_items else {}

            print(f"Ground Truth Functions:")
            if gt_funcs:
                for file_path, funcs in gt_funcs.items():
                    print(f"  {file_path}:")
                    for func in sorted(funcs):
                        print(f"    - {func}")
            else:
                print("  (none)")

            print(f"Predicted Functions:")
            if pred_funcs:
                for file_path, funcs in pred_funcs.items():
                    print(f"  {file_path}:")
                    for func in sorted(funcs):
                        marker = "✓" if file_path in gt_funcs and func in gt_funcs[file_path] else "✗"
                        print(f"    {marker} {func}")
            else:
                print("  (none)")

            # Flatten for metrics calculation
            pred_flat = set()
            gt_flat = set()
            for file_path, funcs in pred_funcs.items():
                for func in funcs:
                    pred_flat.add(f"{file_path}::{func}")
            for file_path, funcs in gt_funcs.items():
                for func in funcs:
                    gt_flat.add(f"{file_path}::{func}")

            intersection = pred_flat & gt_flat
            precision = len(intersection) / len(pred_flat) if pred_flat else 0.0
            recall = len(intersection) / len(gt_flat) if gt_flat else 0.0
            contains_gt = gt_flat.issubset(pred_flat)

        elif evaluation_level == "line":
            pred_lines = pred_items if pred_items else {}
            gt_lines = gt_items

            print(f"Ground Truth Lines:")
            for file_path, lines in gt_lines.items():
                print(f"  {file_path}: {sorted(lines)}")

            print(f"Predicted Lines:")
            for file_path, lines in pred_lines.items():
                print(f"  {file_path}: {sorted(lines)}")
                if file_path in gt_lines:
                    correct_lines = lines & gt_lines[file_path]
                    if correct_lines:
                        print(f"    ✓ Correct: {sorted(correct_lines)}")
                    incorrect_lines = lines - gt_lines[file_path]
                    if incorrect_lines:
                        print(f"    ✗ Incorrect: {sorted(incorrect_lines)}")

            # Flatten for metrics calculation
            pred_flat = set()
            gt_flat = set()
            for file_path, lines in pred_lines.items():
                for line in lines:
                    pred_flat.add(f"{file_path}::{line}")
            for file_path, lines in gt_lines.items():
                for line in lines:
                    gt_flat.add(f"{file_path}::{line}")

            intersection = pred_flat & gt_flat
            precision = len(intersection) / len(pred_flat) if pred_flat else 0.0
            recall = len(intersection) / len(gt_flat) if gt_flat else 0.0
            contains_gt = gt_flat.issubset(pred_flat)

        print(f"Metrics:")
        print(f"  - Precision: {precision:.3f} ({len(intersection)}/{len(pred_flat) if evaluation_level != 'file' else len(pred_files)})")
        print(f"  - Recall: {recall:.3f} ({len(intersection)}/{len(gt_flat) if evaluation_level != 'file' else len(gt_files)})")
        print(f"  - Contains GT: {'Yes' if contains_gt else 'No'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate localization accuracy from generated patches')
    parser.add_argument('--pred_file', type=str, required=True,
                       help='Path to the predictions JSON file containing model_patch field')
    parser.add_argument('--dataset', type=str, default='princeton-nlp/SWE-bench_Verified',
                       help='Dataset name to load ground truth from')
    parser.add_argument('--level', type=str, choices=['all','file', 'function', 'line'],
                       default='all', help='Evaluation level')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed analysis for first few examples')
    parser.add_argument('--detailed-limit', type=int, default=5,
                       help='Number of examples to show in detailed analysis')

    parser.add_argument('--output_file', type=str, help='Output file to save evaluation results')

    args = parser.parse_args()

    # Check if predictions file exists
    if not Path(args.pred_file).exists():
        print(f"Error: Predictions file '{args.pred_file}' not found!")
        return

    if args.level == 'all':
        levels = ['file', 'function', 'line']
    else:
        levels = [args.level]
    
    global REPOS_DIR
    with tempfile.TemporaryDirectory() as tmp_repos_dir:
        REPOS_DIR = Path(tmp_repos_dir)
        outputs = []
        for level in levels:
            print(f"Evaluating at {level} level...")
            print("Loading predictions...")
            predictions = load_predictions(args.pred_file, level, args.dataset)
            print(f"Loaded predictions for {len(predictions)} instances")

            print("Loading ground truth...")
            ground_truth = load_ground_truth(predictions, args.dataset, level)
            print(f"Loaded ground truth for {len(ground_truth)} instances")

            print("Calculating metrics...")
            metrics = calculate_metrics(predictions, ground_truth, evaluation_level=level)
            print(f"\n{'='*60}")
            print(f"EVALUATION RESULTS - {level.upper()} LEVEL")
            print('='*60)
            print(f"Total Problems Evaluated: {metrics['total_problems']}")
            print(f"Precision: {metrics['precision']:.3f} ({metrics['precision']:.1%})")
            print(f"Recall: {metrics['recall']:.3f} ({metrics['recall']:.1%})")
            print(f"Contains GT: {metrics['contains_gt']:.1f}%")
            outputs.append({
                "level": level,
                "metrics": metrics,
            })

            # Print detailed analysis for each level if requested
            if args.detailed:
                print_detailed_analysis(predictions, ground_truth, level, args.detailed_limit)

        print(f'\n\n########')
        for level_dict in outputs:
            print(level_dict['level'])
            p = level_dict["metrics"]["precision"]
            r = level_dict["metrics"]["recall"]
            print(f'- F1: {2 * p * r / (p + r):.1%}')
            # print(f'- Contains GT: {level_dict["metrics"]["contains_gt"]:.1f}%')
            print(f'- Precision: {p:.1%}')
            print(f'- Recall: {r:.1%}')
        print(f'########')
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(outputs, f, indent=4)
                print(f"Evaluation results saved to {args.output_file}")
            
            print(f'Evaluation results saved to {args.output_file}')
            

if __name__ == '__main__':
    main()
