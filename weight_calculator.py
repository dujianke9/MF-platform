"""Operator Frequency Statistics and Weight Calculation Module"""

import math
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter
from config import SMOOTH


def count_operator_frequency(formulas_data: List[Dict]) -> Counter:
    """Count operator frequency across all formulas"""
    counter = Counter()
    for item in formulas_data:
        operators = item.get("operators", [])
        counter.update(operators)
    return counter


def compute_weights(freq_counter: Counter, mode: str = "log_inv", smooth: float = SMOOTH) -> pd.DataFrame:
    """
    Compute operator weights

    mode:
      - log_inv:  w = log((total + smooth)/(freq + smooth)) + 1
      - inv:      w = 1/(freq + smooth)
      - sqrt_inv: w = 1/sqrt(freq + smooth)
    """
    total = sum(freq_counter.values())
    records = []
    for op, freq in freq_counter.most_common():
        if mode == "log_inv":
            w = math.log((total + smooth) / (freq + smooth)) + 1
        elif mode == "inv":
            w = 1.0 / (freq + smooth)
        elif mode == "sqrt_inv":
            w = 1.0 / math.sqrt(freq + smooth)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        records.append({"op": op, "freq": freq, "weight": w})

    df = pd.DataFrame(records)
    # Normalize weights so that the mean equals 1
    if len(df) > 0 and df["weight"].mean() > 0:
        df["weight_normalized"] = df["weight"] / df["weight"].mean()
    else:
        df["weight_normalized"] = df["weight"]
    return df


def compute_all_weights(freq_counter: Counter) -> Dict[str, pd.DataFrame]:
    """Compute weights for all three modes"""
    return {
        "log_inv": compute_weights(freq_counter, "log_inv"),
        "inv": compute_weights(freq_counter, "inv"),
        "sqrt_inv": compute_weights(freq_counter, "sqrt_inv"),
    }
