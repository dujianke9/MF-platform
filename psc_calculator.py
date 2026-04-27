"""
PSC (Physical Structure Consistency) Calculation Module
Consistency evaluation based on physical structure blocks
"""

from typing import List, Dict, Any
from llm_utils import extract_physical_structures


def safe_div(a: float, b: float) -> float:
    """Safe division"""
    return a / b if b != 0 else 0.0


def compute_psc(true_structures: List[Dict], pred_structures: List[Dict]) -> Dict[str, Any]:
    """
    Compute PSC score

    Args:
        true_structures: Physical structure block list of the original formula
                         Each item format: {"name": ..., "fragment": ..., "meaning": ...}
        pred_structures: Physical structure block list of the regression formula

    Returns:
        Dictionary containing PSC score and intermediate steps
    """
    # Build mapping dictionaries using name field for set operations
    true_map = {s["name"]: s for s in true_structures}
    pred_map = {s["name"]: s for s in pred_structures}

    true_names = set(true_map.keys())
    pred_names = set(pred_map.keys())

    overlap_names    = true_names & pred_names
    missing_names    = true_names - pred_names
    extra_names      = pred_names - true_names

    n_true    = len(true_names)
    n_pred    = len(pred_names)
    n_overlap = len(overlap_names)

    # Recall: common structure blocks / original formula structure blocks
    psc_recall = safe_div(n_overlap, n_true)
    # Precision: common structure blocks / regression formula structure blocks
    psc_prec   = safe_div(n_overlap, n_pred)
    # F1 harmonic mean (equivalent to PSC)
    psc_f1     = safe_div(2 * psc_recall * psc_prec, psc_recall + psc_prec)
    psc        = safe_div(2 * n_overlap, n_true + n_pred)

    return {
        # Complete object lists for page display of fragment / meaning
        "S_true"             : sorted(true_structures,  key=lambda x: x["name"]),
        "S_pred"             : sorted(pred_structures,  key=lambda x: x["name"]),

        # Intersection / missing / extra, also preserving complete objects
        "common_structures"  : sorted([true_map[n] for n in overlap_names],  key=lambda x: x["name"]),
        "missing_structures" : sorted([true_map[n] for n in missing_names],  key=lambda x: x["name"]),
        "extra_structures"   : sorted([pred_map[n] for n in extra_names],    key=lambda x: x["name"]),

        "counts": {
            "n_true"  : n_true,
            "n_pred"  : n_pred,
            "n_common": n_overlap,
        },
        "PSC_recall": round(psc_recall, 6),
        "PSC_prec"  : round(psc_prec,   6),
        "PSC_F1"    : round(psc_f1,     6),
        "PSC"       : round(psc,        6),
    }


def compute_psc_with_llm(
    true_formula: str,
    pred_formula: str,
    context: str = ""
) -> dict:
    """
    Compute PSC after extracting physical structure blocks using LLM

    Args:
        true_formula: Original formula text
        pred_formula: Regression formula text
        context:      Physical background description (optional)

    Returns:
        Dictionary containing PSC score and intermediate steps
    """
    true_structures = extract_physical_structures(true_formula, context=context)
    pred_structures = extract_physical_structures(pred_formula, context=context)

    result = compute_psc(true_structures, pred_structures)
    result["true_formula"] = true_formula
    result["pred_formula"] = pred_formula
    return result
