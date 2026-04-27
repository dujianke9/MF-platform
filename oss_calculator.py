"""
OSS (Operator Structure Similarity) Calculation Module (Enhanced Version)
Takes two formula strings as input and returns detailed intermediate results, including:
- Operator list and distribution
- Wasserstein-1 distance
- Operator overlap
- OSS value
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import sympy
from sympy import Basic, Symbol, Number, Add, Mul, Pow
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

# ---------- Operator Frequency and Weights ----------
OP_FREQ = {
    "/": 9743,
    "+": 8339,
    "*": 3148,
    "^": 3120,
    "-": 3901,
    "partial": 2074,
    "int": 800,
    "sqrt": 690,
    "ln": 317,
    "cos": 297,
    "sin": 256,
    "exp": 147,
    "log": 143,
    "tan": 60,
    "in": 49,
    "cosh": 20,
    "sec": 8,
    "oint": 7,
}

def build_sqrt_inv_weights(op_freq, smooth=1.0, scale_to_mean_1=True):
    """Compute sqrt_inv weights"""
    weights = {}
    for op, f in op_freq.items():
        w = 1.0 / math.sqrt(float(f) + smooth)
        weights[op] = w
    if scale_to_mean_1:
        mean_w = sum(weights.values()) / len(weights)
        weights = {k: (v / mean_w if mean_w > 0 else v) for k, v in weights.items()}
    return weights

def normalize_weights_to_unit_interval(weights: Dict[str, float], default: float = 0.5) -> Dict[str, float]:
    """Linearly scale weights to [0,1]"""
    vals = list(weights.values())
    min_w, max_w = min(vals), max(vals)
    if max_w == min_w:
        return {k: default for k in weights}
    return {k: (v - min_w) / (max_w - min_w) for k, v in weights.items()}

# Global semantic axis (operator → [0,1] weight)
SEMANTIC_AXIS = normalize_weights_to_unit_interval(build_sqrt_inv_weights(OP_FREQ))

# ---------- Parsing Helper Functions ----------
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

def normalize_symbol_name(name: str) -> str:
    name = str(name).strip()
    name = name.replace(" ", "_")
    name = name.replace("-", "_")
    return name

def get_variable_names(variable_names: Optional[List[str]] = None) -> List[str]:
    if variable_names is None:
        return []
    return [normalize_symbol_name(v) for v in variable_names]

def preprocess_expression_text(expr_text: str, variable_names: List[str] = None) -> str:
    if not expr_text:
        return ""
    if variable_names is None:
        variable_names = []
    text = str(expr_text).strip()
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("import "):
            continue
        lines.append(s)
    processed_lines = []
    for line in lines:
        line = line.replace("np.", "").replace("math.", "").replace("^", "**").replace("ln(", "log(")
        line = re.sub(r"\babs$", "Abs(", line)
        line = re.sub(r"\bmax$", "Max(", line)
        line = re.sub(r"\bmin$", "Min(", line)

        def repl_x_col(m):
            idx = int(m.group(1))
            if idx < len(variable_names):
                return variable_names[idx]
            return f"x{idx}"
        line = re.sub(r"X$\s*:\s*,\s*(\d+)\s*$", repl_x_col, line)

        def repl_xn(m):
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(variable_names):
                return variable_names[idx]
            return f"x{idx+1}"
        line = re.sub(r"\bX(\d+)\b", repl_xn, line)

        for _ in range(5):
            new_line = line
            new_line = new_line.replace("++", "+").replace("+-", "-").replace("-+", "-").replace("--", "+")
            if new_line == line:
                break
            line = new_line
        processed_lines.append(line.strip())
    return "\n".join(processed_lines)

def parse_assignment_lines(expr_text: str) -> str:
    if not expr_text:
        return ""
    lines = [line.strip() for line in expr_text.splitlines() if line.strip()]
    if not lines:
        return ""
    assignments = {}
    last_lhs = None
    last_rhs = None
    for line in lines:
        if "=" in line:
            left, right = line.split("=", 1)
            left = left.strip()
            right = right.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", left):
                assignments[left] = right
                last_lhs = left
                last_rhs = right
            else:
                last_rhs = right
        else:
            last_rhs = line
    if last_rhs is None:
        return ""

    def expand(expr: str, depth: int = 0) -> str:
        if depth > 20:
            return expr
        changed = False
        result = expr
        for var in sorted(assignments.keys(), key=len, reverse=True):
            if var == last_lhs and expr == assignments.get(var):
                continue
            rhs = assignments[var]
            pattern = rf"\b{re.escape(var)}\b"
            def repl(m):
                return f"({rhs})"
            new_result = re.sub(pattern, repl, result)
            if new_result != result:
                result = new_result
                changed = True
        if changed:
            return expand(result, depth + 1)
        return result
    return expand(last_rhs)

def build_local_dict(expr_text: str, variable_names: List[str] = None) -> Dict[str, Any]:
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr_text or ""))
    local_dict = {
        "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
        "sqrt": sympy.sqrt, "log": sympy.log, "exp": sympy.exp,
        "Abs": sympy.Abs, "Max": sympy.Max, "Min": sympy.Min,
        "pi": sympy.pi,
    }
    if variable_names:
        for tok in variable_names:
            if tok not in local_dict:
                local_dict[tok] = Symbol(tok)
    for tok in tokens:
        if tok not in local_dict:
            local_dict[tok] = Symbol(tok)
    return local_dict

def parse_equation_to_sympy(expr_text: str, variable_names: List[str] = None) -> Optional[Basic]:
    if not expr_text:
        return None
    var_list = get_variable_names(variable_names)
    text = preprocess_expression_text(expr_text, var_list)
    text = parse_assignment_lines(text)
    if not text:
        return None
    for _ in range(5):
        new_text = text
        new_text = new_text.replace("++", "+").replace("+-", "-").replace("-+", "-").replace("--", "+")
        if new_text == text:
            break
        text = new_text
    try:
        local_dict = build_local_dict(text, var_list)
        expr = parse_expr(
            text,
            local_dict=local_dict,
            transformations=TRANSFORMATIONS,
            evaluate=False,
        )
        return expr
    except Exception:
        return None

# ---------- OSS Core Functions ----------
def extract_operators_from_sympy(expr: Basic) -> List[str]:
    """Extract operator list (with repetition) from a sympy expression"""
    if expr is None:
        return []
    ops: List[str] = []
    def traverse(node: Basic):
        if node is None:
            return
        if isinstance(node, (Symbol, Number)):
            return
        if isinstance(node, Add):
            ops.append('+')
        elif isinstance(node, Mul):
            has_div = False
            for arg in node.args:
                if isinstance(arg, Pow):
                    base, exp = arg.as_base_exp()
                    try:
                        if exp == -1:
                            has_div = True
                            break
                    except Exception:
                        pass
            ops.append('/' if has_div else '*')
        elif isinstance(node, Pow):
            base, exp = node.as_base_exp()
            if exp == sympy.Rational(1, 2):
                ops.append('sqrt')
            else:
                ops.append('^')
        elif node.func == sympy.exp:
            ops.append('exp')
        elif node.func == sympy.log:
            ops.append('log')
        elif node.func == sympy.sin:
            ops.append('sin')
        elif node.func == sympy.cos:
            ops.append('cos')
        elif node.func == sympy.tan:
            ops.append('tan')
        elif hasattr(sympy, "cosh") and node.func == sympy.cosh:
            ops.append('cosh')
        elif hasattr(sympy, "sec") and node.func == sympy.sec:
            ops.append('sec')
        if hasattr(node, "args"):
            for arg in node.args:
                if isinstance(arg, Basic):
                    traverse(arg)
    traverse(expr)
    return ops

def compute_operator_distribution(expr: Basic, semantic_axis: Dict[str, float]) -> Dict[float, float]:
    """Compute operator distribution on the 1D semantic axis (key: semantic value, value: probability)"""
    operators = extract_operators_from_sympy(expr)
    if not operators:
        return {}
    counts = Counter(operators)
    total = sum(counts.values())
    if total <= 0:
        return {}
    dist: Dict[float, float] = {}
    for op, c in counts.items():
        p = c / total
        x = semantic_axis.get(op, 0.5)
        dist[x] = dist.get(x, 0.0) + p
    return dist

def compute_wasserstein_1_distance(dist1: Dict[float, float], dist2: Dict[float, float]) -> float:
    """1D Wasserstein-1 distance"""
    if not dist1 or not dist2:
        return 1.0
    xs = sorted(set(dist1.keys()) | set(dist2.keys()))
    if len(xs) == 1:
        x = xs[0]
        return min(abs(dist1.get(x, 0.0) - dist2.get(x, 0.0)), 1.0)
    def cdf(dist: Dict[float, float], xs: List[float]) -> List[float]:
        out = []
        s = 0.0
        for x in xs:
            s += dist.get(x, 0.0)
            out.append(s)
        return out
    c1 = cdf(dist1, xs)
    c2 = cdf(dist2, xs)
    w1 = 0.0
    for i in range(len(xs) - 1):
        dx = xs[i + 1] - xs[i]
        w1 += abs(c1[i] - c2[i]) * dx
    return min(max(w1, 0.0), 1.0)

def compute_operator_overlap(dist1: Dict[float, float], dist2: Dict[float, float]) -> float:
    """Operator distribution overlap"""
    if not dist1 or not dist2:
        return 0.0
    xs = set(dist1.keys()) | set(dist2.keys())
    return sum(min(dist1.get(x, 0.0), dist2.get(x, 0.0)) for x in xs)

def compute_oss(expr_true: Basic, expr_pred: Basic, semantic_axis: Dict[str, float]) -> float:
    """Internal OSS computation (accepts sympy expressions)"""
    if expr_true is None or expr_pred is None:
        return 0.0
    dist_true = compute_operator_distribution(expr_true, semantic_axis)
    dist_pred = compute_operator_distribution(expr_pred, semantic_axis)
    if not dist_true or not dist_pred:
        return 0.0
    w1 = compute_wasserstein_1_distance(dist_true, dist_pred)
    o = compute_operator_overlap(dist_true, dist_pred)
    return (1.0 - w1) * o

# ---------- Enhanced External Interface ----------
def compute_oss_details(true_expr: Basic, pred_expr: Basic,
                        variable_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute OSS between two formula strings and return all intermediate results.
    Note: Input should be formula strings already simplified by sympy.expand().

    Args:
        expr_true_str: String expression of the true formula (after simplification)
        expr_pred_str: String expression of the predicted formula (after simplification)
        variable_names: Optional list of variable names

    Returns:
        dict: Intermediate results including operator lists, distributions, W1 distance, overlap, OSS value, etc.
    """
    # # Step 1: Parse strings into sympy expressions
    # true_expr = parse_equation_to_sympy(expr_true_str, variable_names)
    # pred_expr = parse_equation_to_sympy(expr_pred_str, variable_names)

    # # Step 2: Handle parsing failures
    # if true_expr is None or pred_expr is None:
    #     error_msg = []
    #     if true_expr is None:
    #         error_msg.append("True formula parsing failed")
    #     if pred_expr is None:
    #         error_msg.append("Predicted formula parsing failed")
    #     return {
    #         "error": "; ".join(error_msg),
    #         "true_operators_list": [],
    #         "true_operator_counts": {},
    #         "true_operator_freq": {},
    #         "true_semantic_distribution": {},
    #         "pred_operators_list": [],
    #         "pred_operator_counts": {},
    #         "pred_operator_freq": {},
    #         "pred_semantic_distribution": {},
    #         "wasserstein_distance": 1.0,
    #         "operator_overlap": 0.0,
    #         "oss": 0.0,
    #     }

    # Step 3: Extract operator list (use parsed expression directly, no need to simplify again)
    true_ops_list = extract_operators_from_sympy(true_expr)
    pred_ops_list = extract_operators_from_sympy(pred_expr)

    # Step 4: Operator counting and frequency
    true_counts = Counter(true_ops_list)
    pred_counts = Counter(pred_ops_list)
    true_total = len(true_ops_list)
    pred_total = len(pred_ops_list)

    true_freq = {op: cnt / true_total for op, cnt in true_counts.items()} if true_total > 0 else {}
    pred_freq = {op: cnt / pred_total for op, cnt in pred_counts.items()} if pred_total > 0 else {}

    # Step 5: Distribution on semantic axis
    true_sem_dist = compute_operator_distribution(true_expr, SEMANTIC_AXIS)
    pred_sem_dist = compute_operator_distribution(pred_expr, SEMANTIC_AXIS)

    # Step 6: Compute W1 distance and overlap
    w1 = compute_wasserstein_1_distance(true_sem_dist, pred_sem_dist)
    overlap = compute_operator_overlap(true_sem_dist, pred_sem_dist)

    # Step 7: Compute OSS
    oss = (1.0 - w1) * overlap

    return {
        "error": None,
        "true_operators_list": true_ops_list,
        "true_operator_counts": dict(true_counts),
        "true_operator_freq": true_freq,
        "true_semantic_distribution": true_sem_dist,
        "pred_operators_list": pred_ops_list,
        "pred_operator_counts": dict(pred_counts),
        "pred_operator_freq": pred_freq,
        "pred_semantic_distribution": pred_sem_dist,
        "wasserstein_distance": w1,
        "operator_overlap": overlap,
        "oss": oss,
    }
