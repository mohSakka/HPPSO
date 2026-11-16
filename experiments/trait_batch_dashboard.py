"""Serve saved trial progress and convergence for one batched trait experiment."""
import argparse
from functools import lru_cache
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from statistics import median
import time

import psutil

from experiments.trait_live_progress import log_tail, runner_active


@lru_cache(maxsize=256)
def read_trial(path, modified):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def snapshot(folder):
    plan = json.loads((folder / 'plan.json').read_text(encoding='utf-8'))
    cfg = plan['config']
    entries = plan['protocol']['groups'][cfg['groups'][0]]
    functions = []
    active_seeds = []
    for name in cfg['functions']:
        trials = []
        failures = 0
        for seed in cfg['seeds']:
            file = folder / 'trials' / name / f'seed_{seed:03d}.json.gz'
            if not file.exists():
                progress_file = file.with_suffix('.progress.json')
                try:
                    progress = json.loads(progress_file.read_text())
                    active_seeds.append(dict(progress, name=cfg.get('function_display_names', {}).get(name, name)))
                except (OSError, ValueError):
                    pass
                continue
            run = read_trial(file, file.stat().st_mtime_ns)
            if run['status'] == 'budget_reached':
                trials.append(run)
            else:
                failures += 1
        curves = []
        final_costs = []
        if trials:
            for entry in entries:
                columns = [r['variant_ids'].index(entry['variant']) for r in trials]
                ys = [median(r['trace_best'][i][c] for r, c in zip(trials, columns))
                      for i in range(len(trials[0]['trace_evaluations']))]
                curves.append(ys)
                scores = [r['best_scores'][c] for r, c in zip(trials, columns)
                          if r['best_scores'][c] is not None]
                final_costs.append(dict(median=median(scores) if scores else None,
                                        minimum=min(scores) if scores else None,
                                        seeds=len(scores)))
        functions.append(dict(name=cfg.get('function_display_names', {}).get(name, name),
                              done=len(trials), failed=failures,
                              x=trials[0]['trace_evaluations'] if trials else [],
                              curves=curves, final_costs=final_costs))
    running = runner_active(folder, 'runner.pid', 'experiments.trait_study')
    workers = []
    if running:
        try:
            runner = psutil.Process(int((folder / 'runner.pid').read_text()))
            for worker in runner.children():
                try:
                    cpu = worker.cpu_times()
                    workers.append(dict(pid=worker.pid, cpu=cpu.user + cpu.system))
                except psutil.Error:
                    continue
        except (OSError, ValueError, psutil.Error):
            pass
    return dict(functions=functions, seeds=len(cfg['seeds']), budget=cfg['budget'],
                labels=[e['label'] for e in entries], population=cfg['population'],
                running=running, workers=workers, sampled_at=time.time(),
                active_seeds=[seed for seed in active_seeds if seed['pid'] in {w['pid'] for w in workers}],
                log=log_tail(folder / 'run.log') + log_tail(folder / 'errors.log'))


PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HPPSO openness ablation progress</title>
<style>
body{font:16px system-ui;background:#f2f5f8;color:#182638;margin:24px auto;padding:0 16px;max-width:1100px}
section{background:white;border:1px solid #d9e2ea;border-radius:10px;padding:16px;margin:12px 0}
progress{width:100%;accent-color:#007970;height:16px}h2{font-size:1.1rem}
#panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,450px),1fr));gap:12px}
#panels section{margin:0}svg{width:100%;display:block}pre{white-space:pre-wrap;font-size:13px}
table{width:100%;border-collapse:collapse;font-size:.85rem;font-variant-numeric:tabular-nums}
caption{text-align:left;font-weight:600;margin:8px 0}th,td{padding:5px 3px;border-bottom:1px solid #e2e8ef;text-align:right}
th:first-child{text-align:left}
</style>
<h1>HPPSO openness ablation</h1>
<section><strong id="status">Waiting for the run</strong><p id="details"></p><progress id="bar"></progress>
<p id="activity" aria-live="polite"></p>
<ul id="active-seeds"></ul>
<p id="legend"></p><small>Curves update after each saved seed and show median best-so-far objective. Objective axis uses a signed log transform to show zero and negative values.</small></section>
<div id="panels"></div><section><h2>Recent activity</h2><pre id="log"></pre></section>
<script>
const el=id=>document.getElementById(id), colors=['#0069a8','#bb5200','#007c59','#8236a8','#606975'];
let previousSample=null;
const fmt=x=>Number(x.toPrecision(3)).toLocaleString('en',{maximumSignificantDigits:3});
function plot(f,budget){
  const t=y=>Math.sign(y)*Math.log1p(Math.abs(y));
  const inv=y=>Math.sign(y)*Math.expm1(Math.abs(y));
  const vals=f.curves.flat().map(t);
  let lo=Math.min(...vals),hi=Math.max(...vals);
  if(lo===hi){lo-=.1;hi+=.1}
  const xp=x=>78+390*x/budget, yp=y=>170-150*(y-lo)/(hi-lo);
  let s='<svg viewBox="0 0 480 210" role="img" aria-label="Median best objective versus evaluations">';
  for(let i=0;i<=4;i++){
    const y=lo+(hi-lo)*i/4, py=yp(y), x=budget*i/4, px=xp(x);
    s+=`<path d="M78 ${py}H468 M${px} 20V170" stroke="#e2e8ef"/><text x="73" y="${py+4}" text-anchor="end">${fmt(inv(y))}</text><text x="${px}" y="186" text-anchor="middle">${fmt(x/1000)}k</text>`;
  }
  f.curves.forEach((ys,i)=>s+=`<polyline fill="none" stroke="${colors[i]}" stroke-width="1.8" points="${ys.map((y,j)=>xp(f.x[j])+','+yp(t(y))).join(' ')}"/>`);
  return s+'<text x="273" y="206" text-anchor="middle">Objective evaluations</text></svg>';
}
async function tick(){
 try{
  const response=await fetch('/progress.json',{cache:'no-store'});
  if(!response.ok) throw Error(response.status);
  const d=await response.json(), n=d.functions.reduce((a,f)=>a+f.done,0), failed=d.functions.reduce((a,f)=>a+f.failed,0), total=d.seeds*d.functions.length;
  const state=n===total?'Complete':d.running?'Running':'Stopped or waiting';
  el('status').textContent=`${state} · ${n}/${total} successful seed–function batches · ${n*d.labels.length}/${total*d.labels.length} variant runs`+(failed?` · ${failed} failed batches`:'');
  el('details').textContent=`${d.budget.toLocaleString()} evaluations per variant and seed · ${d.population} particles`;
  el('bar').max=total;el('bar').value=n;
  const previousWorkers=new Map(previousSample?.workers.map(w=>[w.pid,w.cpu])??[]);
  const busy=d.workers.filter(w=>previousWorkers.has(w.pid)&&w.cpu-previousWorkers.get(w.pid)>.1).length;
  el('activity').textContent=d.running
    ? (previousSample?`${busy} workers used CPU since the last refresh. `:'Sampling worker activity. ')
      +'The batch count and curves update when a seed finishes.'
    : '';
  previousSample=d;
  el('active-seeds').replaceChildren(...d.active_seeds.map(seed=>{
    const item=document.createElement('li'), bar=document.createElement('progress');
    const percent=(100*seed.evaluations/d.budget).toFixed(1);
    item.textContent=`${seed.name}, seed ${seed.seed} · ${seed.evaluations.toLocaleString()} / ${d.budget.toLocaleString()} evaluations per variant (${percent}%)`;
    bar.max=d.budget;bar.value=seed.evaluations;item.append(bar);
    return item;
  }));
  el('legend').replaceChildren(...d.labels.map((label,i)=>{const e=document.createElement('span');e.textContent='● '+label+'   ';e.style.color=colors[i];return e}));
  el('panels').replaceChildren(...d.functions.map(f=>{
    const card=document.createElement('section'), title=document.createElement('h2'), bar=document.createElement('progress');
    title.textContent=`${f.name} · ${f.done}/${d.seeds} seeds`+(f.failed?` (${f.failed} failed)`:'');
    bar.max=d.seeds;bar.value=f.done;card.append(title,bar);
    if(f.done){
      card.insertAdjacentHTML('beforeend',plot(f,d.budget));
      const table=document.createElement('table');
      table.createCaption().textContent='Final cost across completed seeds';
      const head=table.createTHead().insertRow();
      for(const label of ['Variant','Median','Minimum','Seeds']){
        const cell=document.createElement('th');
        cell.scope='col';cell.textContent=label;head.append(cell);
      }
      const body=table.createTBody();
      const cost=value=>value===null?'—':value===0?'0':Number(value.toPrecision(8)).toString();
      f.final_costs.forEach((stats,i)=>{
        const row=body.insertRow(), label=document.createElement('th');
        label.scope='row';label.textContent=d.labels[i];label.style.color=colors[i];row.append(label);
        for(const value of [cost(stats.median),cost(stats.minimum),stats.seeds]){
          row.insertCell().textContent=value;
        }
      });
      card.append(table);
    }
    return card;
  }));
  el('log').textContent=d.log.join('\n');
 }catch{el('status').textContent='Progress feed unavailable; retrying';}
 finally{setTimeout(tick,5000);}
}
tick();
</script></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--port', type=int, default=8800)
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                payload, kind = PAGE, 'text/html'
            elif self.path == '/progress.json':
                try:
                    payload = json.dumps(snapshot(args.output), allow_nan=False)
                except (OSError, ValueError):
                    self.send_error(503)
                    return
                kind = 'application/json'
            else:
                self.send_error(404)
                return
            content = payload.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', kind + '; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        server.serve_forever()


if __name__ == '__main__':
    main()
