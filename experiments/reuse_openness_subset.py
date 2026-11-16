"""Reuse completed ablation trials when narrowing the selected function set.

Original files stay intact. Imported records retain their execution fingerprint
under reused_from; fingerprint identifies their destination analysis plan.
"""
import argparse
import json
from pathlib import Path
import time

from experiments.openness_ablation import (
    load_config, fingerprint, jobs, result_path,
)
from experiments.final_benchmark import read_result, write_result
from hppso.algorithms.openness_ablation import VARIANTS


def reuse_subset(source, destination, cfg):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Choose a new destination directory")
    old_plan = json.loads((source / "plan.json").read_text())
    previous = old_plan["config"]
    if not set(cfg["functions"]) <= set(previous["functions"]):
        raise ValueError("Destination must select a subset of the source functions")
    if {k: v for k, v in cfg.items() if k != "functions"} != {
            k: v for k, v in previous.items() if k != "functions"}:
        raise ValueError("Only the function selection may change")
    # This also checks the original runner, algorithm and benchmark source bytes.
    if fingerprint(previous) != old_plan["fingerprint"]:
        raise ValueError("Source code no longer matches the original execution plan")
    digest = fingerprint(cfg)
    destination.mkdir(parents=True)
    provenance = dict(source_directory=str(source), execution_fingerprint=old_plan["fingerprint"],
        reason="Identical code, settings and instances; narrowing function selection",
        imported_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    plan = dict(config=cfg, variants=VARIANTS, fingerprint=digest,
        numpy_version=old_plan["numpy_version"], python_version=old_plan["python_version"],
        imported_trials=provenance)
    (destination / "plan.json").write_text(json.dumps(plan, indent=2))
    definitions = json.loads((source / "problem_definitions.json").read_text())
    (destination / "problem_definitions.json").write_text(json.dumps(
        {name: definitions[name] for name in cfg["functions"]}, indent=2))
    count = 0
    for job in jobs(cfg):
        original = result_path(source, job)
        if not original.exists():
            continue
        record = read_result(original)
        if record["fingerprint"] != old_plan["fingerprint"]:
            raise ValueError(f"Unexpected source fingerprint: {original}")
        if record["status"] != "budget_reached" or record["evaluations"] != cfg["budget"]:
            continue
        record["reused_from"] = dict(provenance, path=str(original),
                                     earlier_reuse=record.get("reused_from"))
        record["fingerprint"] = digest
        write_result(result_path(destination, job), record)
        count += 1
    (destination / "subset_import.json").write_text(json.dumps(dict(provenance,
        imported_trials=count, expected_trials=len(list(jobs(cfg)))), indent=2))
    print(f"Reused {count} completed trials; {len(list(jobs(cfg)))-count} remain", flush=True)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    reuse_subset(args.source, args.output, load_config(args.config))
