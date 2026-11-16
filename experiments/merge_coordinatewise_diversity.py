"""Report coordinatewise HPPSO against the original diversity controls."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.diversity_study import LABELS, report
from experiments.final_benchmark import read_result, write_result
from experiments.trait_study import jobs, result_path


def merge(ablation, original, output):
    plan = json.loads((ablation / "plan.json").read_text())
    archived = json.loads((original / "plan.json").read_text())
    reference = json.loads((original / "reference_plan.json").read_text())
    config = plan["config"]
    expected = {k: v for k, v in config.items()
                if k not in ("coordinatewise_hppso_draws", "reuse_pso_controls")}
    if not config.get("coordinatewise_hppso_draws") or archived["config"] != expected:
        raise ValueError("Diversity protocols do not match")
    output.mkdir(parents=True, exist_ok=True)
    for job in jobs(config):
        fresh = read_result(result_path(ablation, job))
        old = read_result(result_path(original, job))
        pair = fresh["diversity_record"]
        if (fresh["fingerprint"] != plan["fingerprint"]
                or old["fingerprint"] != archived["fingerprint"]
                or fresh["reused_control_fingerprint"] != reference["fingerprint"]
                or fresh["status"] != "budget_reached" or old["status"] != "budget_reached"
                or fresh["evaluations_per_path"] != config["budget"]
                or old["evaluations_per_path"] != config["budget"]
                or not old["reference_ablation_matched"]
                or tuple(old["labels"]) != LABELS or pair["labels"] != ["HPPSO", "PSO"]
                or old["metrics"] != pair["metrics"]
                or old["evaluations"] != pair["evaluations"]):
            raise ValueError(f"Incompatible saved records for {job}")
        for field in ("diversity", "best_errors"):
            values = np.asarray(old[field])
            updated = np.asarray(pair[field])
            np.testing.assert_allclose(values[:, 1], updated[:, 1], rtol=1e-12, atol=1e-10)
            values[:, 0] = updated[:, 0]
            old[field] = values.tolist()
        hp = fresh["variant_ids"].index(plan["protocol"]["factorial"]["15"])
        old["best_positions"][0] = fresh["best_positions"][hp]
        for index, key in enumerate(old["variant_ids"]):
            col = fresh["variant_ids"].index(key)
            np.testing.assert_allclose(old["best_positions"][index], fresh["best_positions"][col],
                                       rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(old["best_errors"][-1][index], fresh["best_scores"][col],
                                       rtol=1e-12, atol=1e-10)
        old.pop("batch_elapsed_seconds", None)
        old.update(fingerprint=plan["fingerprint"], original_diversity_fingerprint=archived["fingerprint"])
        write_result(result_path(output, job), old)
    (output / "plan.json").write_text(json.dumps(dict(
        archived, config=config, fingerprint=plan["fingerprint"],
        reference=str(ablation.resolve()), original_diversity=str(original.resolve()),
        original_diversity_fingerprint=archived["fingerprint"],
    ), indent=2), encoding="utf-8")
    report(output)
    audit_path = output / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit.update(hppso_measured_in_ablation=True, reused_controls=list(LABELS[1:]),
                 new_optimization_evaluations=0)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Four-algorithm diversity report complete: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("ablation", "original", "output"):
        parser.add_argument("--" + option, required=True, type=Path)
    args = parser.parse_args()
    merge(args.ablation, args.original, args.output)
