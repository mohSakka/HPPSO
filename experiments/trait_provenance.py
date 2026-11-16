"""Verify an archived scientific source version without executing archived code."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def snapshot_fingerprint(cfg,archive):
    with ZipFile(archive) as z:
        prefix="hppso-server/"
        names=set(z.namelist())
        paths=["experiments/trait_study.py","experiments/trait_plan.py","experiments/trait_batch.py"]
        if prefix+"experiments/trait_functions.py" in names:paths.append("experiments/trait_functions.py")
        paths += ["src/hppso/algorithms/trait_study.py","src/hppso/algorithms/_pso_bounds.py","experiments/final_benchmark.py"]
        paths += sorted(n[len(prefix):] for n in names if n.startswith(prefix+"src/hppso/benchmarks/") and n.endswith(".py"))
        digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode())
        for path in paths:
            digest.update(path.encode());digest.update(z.read(prefix+path).replace(b"\r\n",b"\n"))
        return digest.hexdigest()
