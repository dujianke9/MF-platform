"""
TSS (Tree Structure Similarity) Calculation Module
Structural similarity based on expression trees
"""

import re
from typing import List, Dict, Any, Optional

import sympy
from sympy import Basic, Symbol, Number
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
from zss import Node, simple_distance

from oss_calculator import (
    preprocess_expression_text,
    parse_assignment_lines,
    TRANSFORMATIONS,
)

def safe_parse(expr_text: str, variable_names: List[str] = None) -> Optional[sympy.Basic]:
    """Safely parse expression"""
    if variable_names is None:
        variable_names = []
    try:
        preprocessed = preprocess_expression_text(expr_text, variable_names)
        expanded = parse_assignment_lines(preprocessed)
        if not expanded:
            expanded = preprocessed

        local_dict = {name: Symbol(name) for name in variable_names}
        expr = parse_expr(expanded, transformations=TRANSFORMATIONS, local_dict=local_dict)
        return expr
    except Exception:
        return None


# ---------- Expression Tree Construction ----------
def sympy_to_zss_tree(expr: Basic) -> Node:
    """Convert sympy expression to zss tree node"""
    if isinstance(expr, Number):
        return Node("NUM")
    if isinstance(expr, Symbol):
        return Node("VAR")

    func_name = type(expr).__name__

    node = Node(func_name)
    for arg in expr.args:
        child = sympy_to_zss_tree(arg)
        node.addkid(child)

    return node


def count_tree_nodes(node: Node) -> int:
    """Count tree nodes"""
    count = 1
    for child in node.children:
        count += count_tree_nodes(child)
    return count

def tree_to_string(node: Node, depth: int = 0) -> str:
    """Convert tree to indented string representation"""
    result = "  " * depth + node.label + "\n"
    for child in node.children:
        result += tree_to_string(child, depth + 1)
    return result


def compute_tss(
    true_expr_text: str,
    pred_expr_text: str,
    variable_names: List[str] = None,
) -> Dict[str, Any]:
    """
    Compute TSS score

    Args:
        true_expr_text: Original formula expression text
        pred_expr_text: Regression formula expression text
        variable_names: List of variable names

    Returns:
        Dictionary containing TSS score and intermediate steps
    """
    if variable_names is None:
        variable_names = []

    # Parse expressions
    true_expr = safe_parse(true_expr_text, variable_names)
    pred_expr = safe_parse(pred_expr_text, variable_names)

    if true_expr is None or pred_expr is None:
        return {
            "error": "Failed to parse expression",
            "true_parsed": true_expr is not None,
            "pred_parsed": pred_expr is not None,
            "TSS": 0.0,
        }

    # Build trees
    true_tree = sympy_to_zss_tree(true_expr)
    pred_tree = sympy_to_zss_tree(pred_expr)

    # Get string representation of trees
    true_tree_str = tree_to_string(true_tree)
    pred_tree_str = tree_to_string(pred_tree)

    # Compute tree edit distance
    ted = simple_distance(true_tree, pred_tree)

    # Count tree nodes
    n_true = count_tree_nodes(true_tree)
    n_pred = count_tree_nodes(pred_tree)

    # Compute TSS
    max_size = n_true + n_pred
    if max_size == 0:
        tss = 1.0
    else:
        tss = 1.0 - ted / max_size
        tss = max(0.0, tss)

    return {
        "true_expression": str(true_expr),
        "pred_expression": str(pred_expr),
        "true_tree": true_tree_str,
        "pred_tree": pred_tree_str,
        "true_tree_nodes": n_true,
        "pred_tree_nodes": n_pred,
        "tree_edit_distance": ted,
        "max_possible_distance": max_size,
        "TSS": round(tss, 6),
    }
