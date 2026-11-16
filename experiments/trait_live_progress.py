"""Serve the coordinatewise trait rerun's live progress on localhost."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

import psutil


RESULTS = Path(__file__).resolve().parent / "results"
ABLATION = RESULTS / "trait_study_coordinatewise_reuse13"
DIVERSITY = RESULTS / "diversity_coordinatewise13"
OPENNESS = RESULTS / "openness_ablation_zero_velocity9"
PILOTS = RESULTS / "archived_pilots_pso_zero"
QUEUE = RESULTS / "zero_velocity_rerun_queue"


def log_tail(path, limit=16):
    if not path.exists():
        return []
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 8192))
        return stream.read().decode("utf-8", errors="replace").splitlines()[-limit:]


def runner_active(folder=ABLATION, pid_file="pipeline.pid", signature="run_coordinatewise_studies.ps1"):
    try:
        pid = int((folder / pid_file).read_text())
        return any(signature in arg for arg in psutil.Process(pid).cmdline())
    except (OSError, ValueError, psutil.Error):
        return False


def progress():
    counts = json.loads((ABLATION / "progress-0.json").read_text())
    config = json.loads((ABLATION / "plan.json").read_text())["config"]
    functions = []
    for name in config["functions"]:
        saved = sum((ABLATION / "trials" / name / f"seed_{seed:03d}.json.gz").is_file()
                    for seed in config["seeds"])
        functions.append(dict(name=name, saved=saved, total=len(config["seeds"])))
    logs = log_tail(ABLATION / "pipeline.log")
    merge_logs = log_tail(ABLATION / "diversity_merge.log")
    errors = log_tail(ABLATION / "diversity_merge_error.log", 8)
    if (DIVERSITY / "audit.json").is_file():
        diversity = "Four-algorithm report ready"
    elif errors and any("Error:" in line or "Traceback" in line for line in errors):
        diversity = "Report generation encountered an error"
    elif not runner_active() and (ABLATION / "audit.json").is_file():
        diversity = "Building four-algorithm report"
    else:
        diversity = "Queued after the ablation audit"
    if counts["completed"] < counts["total"]:
        stage = "Running ablation" if runner_active() else "Ablation paused"
    elif not (ABLATION / "audit.json").is_file():
        stage = "Generating ablation reports" if runner_active() else "Ablation report incomplete"
    else:
        stage = "Ablation complete"
    return dict(counts, functions=functions, stage=stage, diversity=diversity, unit="batches",
                messages=logs + merge_logs + errors, reruns=rerun_progress())


def rerun_progress():
    queue_running = runner_active(QUEUE, "runner.pid", "run_zero_velocity_reruns.ps1")
    messages = log_tail(QUEUE / "run.log") + log_tail(QUEUE / "error.log", 8)
    pilot_plan = PILOTS / "plan.json"
    if pilot_plan.is_file():
        jobs = json.loads(pilot_plan.read_text())["jobs"]
        expected = Counter(job["function"] for job in jobs)
        files = list(PILOTS.glob("*/trials/*/*/*/seed_*.json.gz"))
        saved = Counter(f.parents[1].name for f in files)
        stage = "Local PSO pilots"
        finished = (PILOTS / "runs.csv").is_file()
    elif (OPENNESS / "plan.json").is_file():
        plan = json.loads((OPENNESS / "plan.json").read_text())
        expected = {name: len(plan["config"]["seeds"]) * len(plan["variants"])
                    for name in plan["config"]["functions"]}
        files = list(OPENNESS.glob("trials/*/*/seed_*.json.gz"))
        saved = Counter(f.parents[1].name for f in files)
        stage = "Zero-velocity openness"
        finished = (OPENNESS / "report_status.json").is_file()
    else:
        return None
    completed, total = sum(saved.values()), sum(expected.values())
    active = queue_running or runner_active(OPENNESS, "runner.pid", "experiments.openness_ablation")
    if finished:
        stage += " complete"
    elif completed == total:
        stage += " · reports pending"
    elif not active:
        stage += " paused"
    updated = max((f.stat().st_mtime for f in files), default=0)
    return dict(stage=stage, completed=completed, total=total, unit="saved trials",
                updated_utc=datetime.fromtimestamp(updated, timezone.utc).isoformat(),
                functions=[dict(name=name, saved=saved[name], total=count) for name, count in expected.items()],
                messages=messages)


PAGE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HPPSO experiment progress</title>
<style>
  :root { color-scheme: light; font-family: Inter, 'Segoe UI', sans-serif; color: #182638; background: #f2f5f8 }
  body { margin: 0; padding: 28px 16px }
  main { max-width: 960px; margin: auto }
  header { margin: 4px 0 22px }
  h1 { margin: 0; font-size: clamp(1.55rem, 4vw, 2.15rem); letter-spacing: -.035em }
  h2 { margin: 0 0 14px; font-size: 1.04rem }
  p { margin: 8px 0 0; color: #4b5c6e }
  section { background: white; padding: 20px; border: 1px solid #d9e2ea; border-radius: 12px; margin-bottom: 14px }
  .between { display: flex; justify-content: space-between; gap: 12px; align-items: baseline }
  .number { font-size: 1.65rem; font-weight: 700; font-variant-numeric: tabular-nums }
  progress { display: block; width: 100%; height: 18px; accent-color: #006f73; margin: 12px 0 }
  .detail, time { font-size: .9rem; color: #4b5c6e }
  .function { display: grid; grid-template-columns: minmax(120px, 1.35fr) 1fr 3.3rem; align-items: center; gap: 12px; min-height: 33px; font-size: .9rem }
  .function progress { height: 9px; margin: 0 }
  .function span:last-child { text-align: right; font-variant-numeric: tabular-nums }
  pre { margin: 0; font: .82rem/1.55 'Cascadia Mono', Consolas, monospace; background: #142434; color: #e4edf5; padding: 12px; border-radius: 8px; overflow: auto; max-height: 260px }
</style>
<main>
  <header>
    <h1>HPPSO experiment progress</h1>
    <p id="ablationSummary"></p>
  </header>
  <section aria-labelledby="stage">
    <div class="between"><h2 id="stage">Connecting</h2><strong class="number" id="percent">—</strong></div>
    <progress id="overall" aria-label="Overall batch progress" max="130" value="0"></progress>
    <div class="between detail"><span id="count"></span><time id="updated"></time></div>
    <p id="diversity"></p>
  </section>
  <section><h2>Progress by function</h2><div id="functions"></div></section>
  <section><h2>Latest run messages</h2><pre id="messages"></pre></section>
</main>
<script>
  const byId = id => document.getElementById(id);
  async function refresh() {
    try {
      const response = await fetch('/progress.json', {cache: 'no-store'});
      if (!response.ok) throw new Error('Progress unavailable');
      const state = await response.json();
      const p = state.reruns || state;
      byId('ablationSummary').textContent = state.stage + ' · ' + state.completed + ' / ' + state.total + ' batches';
      byId('stage').textContent = p.stage;
      byId('percent').textContent = (100 * p.completed / p.total).toFixed(1) + '%';
      byId('overall').max = p.total;
      byId('overall').value = p.completed;
      byId('count').textContent = p.completed + ' / ' + p.total + ' ' + p.unit;
      byId('updated').textContent = 'Run updated ' + new Date(p.updated_utc).toLocaleTimeString();
      byId('diversity').textContent = 'Diversity · ' + state.diversity;
      byId('functions').replaceChildren(...p.functions.map(f => {
        const row = document.createElement('div');
        row.className = 'function';
        const name = document.createElement('span');
        name.textContent = f.name.replace(/^f\d+_/, '').replaceAll('_', ' ');
        const bar = document.createElement('progress');
        bar.max = f.total;
        bar.value = f.saved;
        bar.setAttribute('aria-label', name.textContent);
        const count = document.createElement('span');
        count.textContent = f.saved + ' / ' + f.total;
        row.append(name, bar, count);
        return row;
      }));
      byId('messages').textContent = p.messages.join('\n');
      byId('messages').scrollTop = byId('messages').scrollHeight;
    } catch {
      byId('stage').textContent = 'Waiting for the progress feed';
    } finally {
      setTimeout(refresh, 3000);
    }
  }
  refresh();
</script>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            content = PAGE.encode("utf-8")
            media_type = "text/html; charset=utf-8"
        elif self.path == "/progress.json":
            try:
                content = json.dumps(progress()).encode("utf-8")
            except (OSError, ValueError):
                self.send_error(503, "Progress is temporarily unavailable")
                return
            media_type = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *_args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8799)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Live progress: http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()
