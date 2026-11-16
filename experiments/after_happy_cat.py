"""Windows controller: finish Happy Cat, stop the old tree, start revised 30D."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def finished_happy_cat(rows):
    return {(r["algorithm"], r["seed"]): r for r in rows
            if r["phase"] == "full" and r["dimension"] == 1000
            and r["function"] == "f15_happy_cat"}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--old-dir",type=Path,required=True)
    parser.add_argument("--new-dir",type=Path,required=True)
    parser.add_argument("--old-pid",type=int,required=True)
    parser.add_argument("--port",type=int,default=8765)
    args=parser.parse_args()
    old=args.old_dir.resolve()
    new=args.new_dir.resolve()
    new.mkdir(parents=True,exist_ok=True)
    status_path=old/"transition_status.json"
    def status(state, **kw):
        status_path.write_text(json.dumps(dict(state=state,updated=time.time(),new_directory=str(new),**kw)),encoding="utf-8")
    # psutil is used only by this local Windows controller, not by optimizers.
    import psutil
    process=psutil.Process(args.old_pid)
    command=" ".join(process.cmdline())
    if "experiments.smoke" not in command or str(args.old_dir).replace("\\","/") not in command.replace("\\","/"):
        raise RuntimeError("Old PID does not identify the expected experiment")
    created=process.create_time()
    expected={(a,s) for a in ("Original PSO","PSO-m","PSO-RIW","HPPSO","GA-MPC","GWO","SHADE","CSA","Sep-CMA-ES") for s in (0,1)}
    journal=old/"trials.jsonl"
    while True:
        rows=[]
        for line in journal.read_text().splitlines():
            try: rows.append(json.loads(line))
            except ValueError: pass
        finished=finished_happy_cat(rows)
        status("waiting_for_happy_cat",completed=len(expected & finished.keys()),total=len(expected),old_pid=args.old_pid)
        if expected <= finished.keys():
            break
        if not process.is_running():
            status("blocked_old_process_stopped",completed=len(expected & finished.keys()),total=len(expected))
            return
        time.sleep(2)
    # Validate identity again immediately before terminating this one process tree.
    process=psutil.Process(args.old_pid)
    if process.create_time()!=created:
        raise RuntimeError("PID identity changed; refusing to stop a different process")
    if "experiments.smoke" not in " ".join(process.cmdline()):
        raise RuntimeError("Experiment command changed")
    status("stopping_old_experiment",completed=18,total=18)
    result=subprocess.run(["taskkill","/PID",str(args.old_pid),"/T","/F"],capture_output=True,text=True)
    if result.returncode:
        status("stop_failed",error=result.stdout+result.stderr)
        return
    # The old server has been killed together with its workers.
    time.sleep(2)
    env=os.environ.copy()
    repo=Path(__file__).resolve().parents[1]
    env["PYTHONPATH"]=str(repo/"src")+os.pathsep+str(repo)
    with (new/"output.log").open("a",encoding="utf-8") as log:
        child=subprocess.Popen([sys.executable,"-u","-m","experiments.smoke",
                "--output-dir",str(new),"--dimensions","30","--workers","2",
                "--port",str(args.port),"--phase","full"],cwd=repo,env=env,
                stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    (new/"process_id.txt").write_text(str(child.pid))
    status("revised_30d_started",completed=18,total=18,new_pid=child.pid,
           happy_cat_failed=sum(r["status"]=="failed" for r in finished.values()))
    print(f"Revised 30D process started: {child.pid}",flush=True)


if __name__=="__main__":
    main()
