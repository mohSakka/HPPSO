"""CEC 2011 real-world optimization problems via minionpy."""

from __future__ import annotations

import numpy as np


def get_cec2011_info(func_num: int):
    """Return (objective, dimension, (lb, ub)) for CEC2011 problem T{func_num}."""
    from minionpy import cec

    suite = cec.CEC2011Functions(function_number=func_num)
    target_dim = suite.dimension
    meta = cec.CEC2011_METADATA[func_num]
    lb, ub = meta[1], meta[2]
    # Original CEC2011 technical report, pp. 7 and 33: 0.6 <= u <= 0.9.
    # MinionPy 1.9.1 incorrectly supplies a negative lower bound for T3.
    if func_num == 3:
        lb, ub = np.array([0.6]), np.array([0.9])

    def wrapper(x):
        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        return float(suite.evaluate(x_arr)[0])

    return wrapper, target_dim, (lb, ub)


def load_cec2011_problems():
    """Load all 22 CEC2011 problems (T1..T22)."""
    problems = []
    for i in range(1, 23):
        try:
            func, dim, bounds = get_cec2011_info(i)
            problems.append({"name": f"T{i}", "function": func, "dimension": dim, "bounds": bounds})
        except Exception as exc:
            print(f"Could not load T{i}: {exc}")

    for problem in problems:
        if problem["name"] == "T3":
            problem["bounds"] = (0.6, 0.9)
            break

    return problems
