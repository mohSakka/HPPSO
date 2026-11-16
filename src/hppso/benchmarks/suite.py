"""Unified registry: 20 classical functions plus ten additional hard problems."""
from .additional import NAMES as ADDITIONAL_NAMES, VERSION, build_additional_suite
from .classical import FUNCTION_NAMES, build_problem_suite

BENCHMARK_NAMES = tuple(FUNCTION_NAMES) + ADDITIONAL_NAMES
SUITE_VERSION = "classical20-" + VERSION


def build_benchmark_suite(dim=30, random_shift=False, fixed_shifts=None,
                          michalewicz_dimension=10, additional_seed=42):
    problems = build_problem_suite(dim, random_shift, fixed_shifts, michalewicz_dimension)
    problems.extend(p.as_problem() for p in build_additional_suite(dim, additional_seed, fixed_shifts=fixed_shifts))
    return problems
