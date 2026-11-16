"""Serve live seed-extension progress from saved trial and worker files."""

from collections import Counter
from functools import lru_cache, partial
import gzip
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time


OUTPUT = Path(__file__).resolve().parent / 'results/cec2011_coordinate_refresh_10_seeds'
PORT = 8797

PAGE = r'''<!doctype html><meta charset="utf-8"><title>CEC2011 progress</title>
<style>body{font:16px system-ui;max-width:900px;margin:35px auto;padding:0 15px}
progress{width:100%;height:26px}td,th{padding:5px 14px;text-align:left}
pre{white-space:pre-wrap}</style>
<h1>CEC2011 · 10-seed benchmark</h1>
<p>Seeds 0–2 are already complete. This page tracks the seven new seeds.</p>
<p id="status" aria-live="polite">Reading saved trials…</p>
<progress id="bar" aria-label="New trials completed" max="1386" value="0"></progress>
<h2>Active trials</h2><pre id="active"></pre>
<h2>Completed by problem</h2><table><thead><tr><th>Problem</th><th>New trials</th></tr></thead>
<tbody id="problems"></tbody></table>
<p><a href="plan.json">Experiment settings</a> · <a href="tables/final_scores.csv">Final scores when ready</a></p>
<script>
const find = id => document.getElementById(id);
async function update() {
  try {
    const response = await fetch('/live.json', {cache:'no-store'});
    if (!response.ok) throw new Error('Progress unavailable');
    const p = await response.json();
    find('status').textContent = p.completed + ' / ' + p.total + ' new trials complete (' +
      (100 * p.completed / p.total).toFixed(1) + '%) · ' + p.failed + ' failed';
    find('bar').value = p.completed;
    find('active').textContent = p.active.map(r => 'T' + r.problem + ' · ' + r.algorithm +
      ' · seed ' + r.seed + ' · ' + r.evaluations + ' / ' + r.budget + ' evaluations').join('\n') ||
      'No recent worker updates.';
    find('problems').replaceChildren(...Object.entries(p.problems).map(([problem, count]) => {
      const row = document.createElement('tr');
      row.insertCell().textContent = 'T' + problem;
      row.insertCell().textContent = count + ' / 63';
      return row;
    }));
  } catch (_) {
    find('status').textContent = 'Waiting for a progress update…';
  }
}
update(); setInterval(update, 5000);
</script>'''


@lru_cache(maxsize=2000)
def trial_status(path, modified):
    with gzip.open(path, 'rt') as stream:
        trial = json.load(stream)
    completed = trial['status'] == 'budget_reached' and trial['evaluations'] == 200000
    return trial['problem'], completed


def snapshot():
    completed = Counter()
    failed = 0
    for path in OUTPUT.glob('trials/T*/*/seed_*.json.gz'):
        try:
            problem, valid = trial_status(path, path.stat().st_mtime_ns)
        except (OSError, ValueError, EOFError):
            continue
        if valid:
            completed[problem] += 1
        else:
            failed += 1
    active = []
    for path in (OUTPUT / 'active').glob('worker_*.json'):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not row['done'] and time.time() - row['updated'] < 120:
            active.append(row)
    return dict(completed=sum(completed.values()), total=1386, failed=failed, active=active,
                problems={p: completed[p] for p in range(1, 23)})


class ProgressHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            content, data = 'text/html', PAGE
        elif self.path == '/live.json':
            content, data = 'application/json', json.dumps(snapshot())
        else:
            return super().do_GET()
        payload = data.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', f'{content}; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    handler = partial(ProgressHandler, directory=str(OUTPUT))
    ThreadingHTTPServer(('127.0.0.1', PORT), handler).serve_forever()
