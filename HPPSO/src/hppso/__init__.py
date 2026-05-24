"""HPPSO: Human Personality Based Particle Swarm Optimization."""

__version__ = "1.0.0"

from .algorithms import HPPSO, HPPSO_Modified, PSO
from .benchmarks import BenchmarkFunctions, build_problem_suite

__all__ = ["HPPSO", "HPPSO_Modified", "PSO", "BenchmarkFunctions", "build_problem_suite"]
