"""Serve progress and convergence traces for the local neural-network benchmark."""

import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from hppso.experiments.train_neural_network import ALGORITHMS, DATASETS


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/nn_pop30_ef3000_seed10_cap2000"
PAGE = Path(__file__).with_suffix(".html")
BUDGET = 3000
SEEDS = range(10)


@lru_cache(maxsize=40)
def read_trials(path, modified_ns, size):
    rows = {}
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break  # The worker may still be writing this line.
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if (row["dataset"] == path.parent.name and row["evaluations"] == BUDGET
                    and row["seed"] in SEEDS and row["algorithm"] in ALGORITHMS):
                rows[(row["algorithm"], row["seed"])] = row
    return list(rows.values())


def dataset_trials(output, dataset):
    path = output / dataset / "trials.jsonl"
    try:
        info = path.stat()
        return read_trials(path, info.st_mtime_ns, info.st_size), info.st_mtime_ns
    except OSError:
        return [], 0


def progress(output):
    exits = {}
    for path in output.glob("*queue.log"):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            dataset, marker, code = line.partition(": exit ")
            if marker and dataset in DATASETS and code.strip().lstrip("-").isdigit():
                exits[dataset] = int(code)
    datasets = []
    newest = None
    newest_time = 0
    per_dataset = len(ALGORITHMS) * len(SEEDS)
    for key, (label, task, count, *_rest) in DATASETS.items():
        rows, updated = dataset_trials(output, key)
        if len(rows) == per_dataset:
            status = "Complete"
        elif exits.get(key, 0):
            status = "Error"
        elif (output / f"{key}.log").exists():
            status = "Started"
        else:
            status = "Queued"
        datasets.append(dict(key=key, label=label, task=task, samples=min(count, 2000),
                             completed=len(rows), total=per_dataset, status=status))
        if rows and updated > newest_time:
            newest_time = updated
            newest = {field: rows[-1][field] for field in
                      ("dataset", "algorithm", "seed", "seconds", "training_loss")}
    return dict(datasets=datasets, algorithms=ALGORITHMS, latest=newest, budget=BUDGET,
                completed=sum(item["completed"] for item in datasets),
                total=len(datasets) * per_dataset)


def handler_for(output):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/":
                self.send_data("text/html", PAGE.read_bytes())
            elif url.path == "/api/progress":
                self.send_json(progress(output))
            elif url.path == "/api/curves":
                params = parse_qs(url.query)
                dataset = params.get("dataset", [""])[0]
                algorithm = params.get("algorithm", [""])[0]
                if dataset not in DATASETS or algorithm not in ALGORITHMS:
                    self.send_error(400, "Unknown dataset or algorithm")
                    return
                rows, _ = dataset_trials(output, dataset)
                curves = [{field: row[field] for field in ("seed", "convergence", "training_loss", "seconds")}
                          for row in rows if row["algorithm"] == algorithm]
                self.send_json(curves)
            else:
                self.send_error(404)

        def send_json(self, data):
            self.send_data("application/json", json.dumps(data, allow_nan=False).encode("utf-8"))

        def send_data(self, media_type, payload):
            self.send_response(200)
            self.send_header("Content-Type", f"{media_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--port", type=int, default=8798)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(args.output.resolve())) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
