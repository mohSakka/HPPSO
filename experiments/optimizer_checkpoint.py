"""Numerical-state snapshots plus verified deterministic evaluation replay.

Python call stacks are not pickled. On recovery the unchanged optimizer is
replayed using journaled fitnesses, restoring its pending calls without invoking
the objective again. Candidate hashes reject divergent trajectories.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import struct
import sys
import time
import numpy as np
from experiments.final_benchmark import BudgetComplete

DTYPE=np.dtype([('digest','V16'),('value','<f8')])


def replace_retry(source,target):
    for attempt in range(200):
        try:os.replace(source,target);return
        except PermissionError:
            if attempt==199:raise
            time.sleep(.05)


def numeric(value,depth=0):
    if depth>8:return None
    if value is None or isinstance(value,(str,bytes,bool,int,float,np.generic,np.ndarray)):return value
    if isinstance(value,(list,tuple)):
        result=[numeric(x,depth+1) for x in value]
        return tuple(result) if isinstance(value,tuple) else result
    if isinstance(value,dict):return {k:numeric(v,depth+1) for k,v in value.items() if isinstance(k,(str,bytes,int,float,tuple))}
    return None


def frames():
    result=[];frame=sys._getframe(1)
    while frame is not None:
        filename=frame.f_code.co_filename.replace('\\','/')
        if '/hppso/algorithms/' in filename:
            state={}
            for key,value in frame.f_locals.items():
                if key=='self':state['optimizer_attributes']=numeric(vars(value))
                elif not callable(value):state[key]=numeric(value)
            result.append(dict(file=filename,function=frame.f_code.co_name,line=frame.f_lineno,locals=state))
        frame=frame.f_back
    return result


class CheckpointObjective:
    def __init__(self,function,job,folder,fingerprint,interval=10000,points=600):
        self.function=function;self.job=job;self.budget=int(job['budget'])
        self.horizon=int(job['schedule_horizon']);self.interval=interval
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        header=dict(format=1,fingerprint=fingerprint,identity={k:job[k] for k in
            ('problem','algorithm','dimension','population','seed','schedule_horizon')},record_bytes=24)
        path=self.folder/'header.json'
        if path.exists():
            if json.loads(path.read_text())!=header:raise ValueError('Checkpoint source or immutable settings changed')
        else:
            temp=path.with_suffix('.tmp')
            with temp.open('w') as f:json.dump(header,f,indent=2);f.flush();os.fsync(f.fileno())
            replace_retry(temp,path)
        self.journal=self.folder/'evaluations.bin'
        if self.journal.exists():
            size=self.journal.stat().st_size
            if size%24:
                with self.journal.open('r+b') as f:f.truncate(size-size%24)
            self.cached=np.fromfile(self.journal,dtype=DTYPE)
        else:self.cached=np.empty(0,dtype=DTYPE)
        if len(self.cached)>self.budget:raise ValueError('Target budget is smaller than the saved journal')
        self.stream=self.journal.open('ab');self.count=0;self.new_evaluations=0
        self.best=float('inf');self.best_position=None;self.nonfinite_candidates=0
        self.trace=[];self.last_snapshot=0
        n=min(points,self.budget)
        self.checkpoints=set(np.unique(np.r_[np.geomspace(1,self.budget,n).astype(int),
                                               np.linspace(1,self.budget,n).astype(int)]).tolist())
        self.started=time.perf_counter();self.last_publish=0
        self.active=None
        self.current_candidate=None
        self.saved_state=None
        if (self.folder/'state.pkl').exists():
            with (self.folder/'state.pkl').open('rb') as f:self.saved_state=pickle.load(f)
            self.last_snapshot=self.saved_state['evaluations']
            if self.last_snapshot>len(self.cached):raise ValueError('Journal is shorter than its committed snapshot')

    def evaluation_progress(self):
        # Extension continues the original schedule; it never rewrites history.
        return min(self.count/self.horizon,1.)

    def publish(self,done=False):
        if self.active is not None:
            data=dict(self.job,evaluations=self.count,replayed_evaluations=min(self.count,len(self.cached)),
                new_evaluations=self.new_evaluations,checkpoint_saved=self.last_snapshot,
                best_score=self.best if np.isfinite(self.best) else None,nonfinite_candidates=self.nonfinite_candidates,
                elapsed_seconds=time.perf_counter()-self.started,updated=time.time(),done=done)
            try:
                temp=self.active.with_suffix('.tmp');temp.write_text(json.dumps(data,allow_nan=False));replace_retry(temp,self.active)
            except OSError:pass
        self.last_publish=time.perf_counter()

    def snapshot(self):
        self.stream.flush();os.fsync(self.stream.fileno())
        state=dict(format=1,evaluations=self.count,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            algorithm_frames=frames(),pending_candidate=self.current_candidate,
            best_position=self.best_position,
            best_score=self.best,nonfinite_candidates=self.nonfinite_candidates,
            restore_method='verified journal replay; no previous objective reevaluation',schedule_horizon=self.horizon)
        temp=self.folder/'state.pkl.tmp'
        with temp.open('wb') as f:pickle.dump(state,f,protocol=5);f.flush();os.fsync(f.fileno())
        replace_retry(temp,self.folder/'state.pkl');self.last_snapshot=self.count

    def __call__(self,x):
        if self.count>=self.budget:raise BudgetComplete
        point=np.ascontiguousarray(x,dtype='<f8')
        self.current_candidate=point.copy()
        digest=hashlib.blake2b(point.tobytes(),digest_size=16).digest()
        if self.count<len(self.cached):
            record=self.cached[self.count]
            if digest!=record['digest'].tobytes():
                raise RuntimeError(f'Checkpoint trajectory diverged before evaluation {self.count+1}; objective was not reevaluated')
            raw=float(record['value'])
        else:
            raw=float(self.function(x));self.new_evaluations+=1
            self.stream.write(struct.pack('<16sd',digest,raw))
        self.count+=1;value=raw
        if not np.isfinite(value):self.nonfinite_candidates+=1;value=float('inf')
        if value<self.best:self.best=value;self.best_position=np.asarray(x,dtype=float).copy()
        if self.saved_state is not None and self.count==self.saved_state['evaluations']:
            if pickle.dumps(random.getstate())!=pickle.dumps(self.saved_state['python_rng']):
                raise RuntimeError('Python RNG state differs from checkpoint')
            old=self.saved_state['numpy_rng'];now=np.random.get_state()
            if old[0]!=now[0] or not np.array_equal(old[1],now[1]) or old[2:]!=now[2:]:
                raise RuntimeError('NumPy RNG state differs from checkpoint')
        if self.count in self.checkpoints and np.isfinite(self.best):self.trace.append([self.count,self.best])
        if self.count>len(self.cached) and self.count%1000==0:
            self.stream.flush();os.fsync(self.stream.fileno())
        if (self.count>len(self.cached) and self.count%self.interval==0) or self.count==self.budget:
            self.snapshot();self.publish()
        elif time.perf_counter()-self.last_publish>=2:self.publish()
        if self.count==self.budget:raise BudgetComplete
        return value

    def close(self):
        self.stream.flush();os.fsync(self.stream.fileno());self.stream.close()
