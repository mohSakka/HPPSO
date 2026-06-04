"""Configuration and naming conventions for paper reproduction."""

from __future__ import annotations

from pathlib import Path

# Repository root (HPPSO package directory).
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
OUTPUT_FIGURES_DIR = ARTIFACTS_DIR / "reproduced_figures"
OUTPUT_TABLES_DIR = ARTIFACTS_DIR / "reproduced_tables"

# Known result file patterns (filenames vary slightly; loaders resolve by keyword).
FILE_PATTERNS = {
    "convergence_30d_main": "all_convergence_histories 30d.pkl",
    "convergence_30d_cmaes": "all_convergence_histories 30d cmaes only.pkl",
    "convergence_1000d": "all_convergence_histories1000.pkl",
    "convergence_1000d_alt": "all_convergence_histories.pkl",
    "best_costs_30d": "best_costs_30d.pkl",
    "shift_vectors_30d": "all_svs30d.pkl",
    "shift_vectors_30d_cmaes": "all_svs 30d cmaes only.pkl",
    "shift_vectors_1000d": "all_svs 1000.pkl",
    "shift_vectors_1000d_alt": "all_svs.pkl",
}

DIMENSIONS = (30, 1000)

# Internal algorithm keys as stored in pickle files -> paper display names.
ALGORITHM_DISPLAY_NAMES = {
    "Original PSO": "PSO",
    "PSO-m": "PSO-M",
    "PSO-RIW": "PSO-RIW",
    "HPPSO_Modified": "HPPSO",
    "GA-MPC": "GA-MPC",
    "GWO": "GWO",
    "SHADE": "SHADE",
    "CSA": "CSA",
    "CMAES": "CMAES",
    "Sep-CMA-ES": "SEP-CMAES",
}

# Paper table column order (sep-CMAES may be missing from result files).
PAPER_ALGORITHM_ORDER_30D = [
    "PSO",
    "PSO-M",
    "PSO-RIW",
    "HPPSO",
    "GA-MPC",
    "GWO",
    "SHADE",
    "CSA",
    "CMAES",
]

PAPER_ALGORITHM_ORDER_1000D = [
    "PSO",
    "PSO-M",
    "PSO-RIW",
    "HPPSO",
    "GA-MPC",
    "GWO",
    "SHADE",
    "CSA",
    "SEP-CMAES",
]

FUNCTION_NAMES = [
    "f1_sphere",
    "f2_schwefel_2_22",
    "f3_zakharov",
    "f4_rosenbrock",
    "f5_powell_sum",
    "f6_rastrigin",
    "f7_ackley",
    "f8_griewank",
    "f9_weierstrass",
    "f10_levy",
    "f11_schwefel_2_26",
    "f12_styblinski_tang",
    "f13_michalewicz",
    "f14_alpine_1",
    "f15_happy_cat",
    "f16_hgbat",
    "f17_bent_cigar",
    "f18_discus",
    "f19_penalized_1",
    "f20_zakharov_2",
]

# Figure 7 / Figure 8 subplot mapping (15 of 20 functions, paper order).
FIGURE_PANEL_FUNCTIONS = [
    ("f1_sphere", "Sphere Function"),
    ("f4_rosenbrock", "Rosenbrock Function"),
    ("f6_rastrigin", "Rastrigin Function"),
    ("f11_schwefel_2_26", "Schwefel 2.26 Function"),
    ("f13_michalewicz", "Michalewicz Function"),
    ("f8_griewank", "Griewank Function"),
    ("f15_happy_cat", "Happy Cat Function"),
    ("f19_penalized_1", "Penalized Function"),
    ("f17_bent_cigar", "Bent Cigar Function"),
    ("f10_levy", "Levy Function"),
    ("f9_weierstrass", "Weierstrass Function"),
    ("f18_discus", "Discus Function"),
    ("f12_styblinski_tang", "Styblinski–Tang Function"),
    ("f7_ackley", "Ackley Function"),
    ("f14_alpine_1", "Alpine Function"),
]

NUM_RUNS = 20
MAX_ITERS = 500
POP_SIZE = 30
CMAES_ALGORITHM_KEY = "CMAES"
HPPSO_ALGORITHM_KEY = "HPPSO_Modified"

# Shift-vector pickle sources (one per dimension; 20 seeds × 20 functions).
SHIFT_VECTOR_SOURCES = {
    30: {
        "pickle": "results/all_svs30d.pkl",
        "description": "30D shift vectors — one vector per function (same across 20 seeds)",
        "exported_csv": "shift_vectors_30D.csv",
    },
    1000: {
        "pickle": "results/all_svs 1000.pkl",
        "description": "1000D shift vectors — one vector per function (same across 20 seeds)",
        "exported_csv": "shift_vectors_1000D.csv",
        "fallback_pickle": "results/all_svs.pkl",
    },
}

# Paper reference values for sanity checks (Table 2 f1 PSO mean).
REFERENCE_VALUES = {
    (30, "f1_sphere", "PSO"): 1.41e4,
}
