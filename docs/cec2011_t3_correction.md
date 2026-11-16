# T3 domain correction

The original CEC2011 technical report specifies T3's one-dimensional search box
as [0.6,0.9] (pages 7 and 33):
https://al-roomi.org/multimedia/CEC_Database/CEC2011/CEC2011_TechnicalReport.pdf

MinionPy 1.9.1 supplies [-0.6,0.9] in its metadata. The prior native-metadata runs
therefore used an incorrect box. The repository's original T3 override was valid;
removing it was an error. All current CEC2011 entry points now override T3 bounds
to [0.6,0.9] without modifying the installed third-party package or its objective.

The latest comparison reruns the 27 T3 trials only, using population 200, seeds
0–2, 1000 iterations and the GA-MPC 200,000-evaluation cap. Every T3 objective call
is checked against the corrected interval. The other 21 problems are copied
unchanged with preserved fingerprints. The new plan records a problem-specific
fingerprint for T3 and the report URL identifying the correction.

```powershell
$env:PYTHONPATH='D:\hppso-git\.deps\cec2011;D:\hppso-git\src'
python -u -m experiments.cec2011_correct_t3 --workers 2
```

Live results and updated ranks/minimum tables: http://127.0.0.1:8769/.
Output: experiments/results/cec2011_pop200_ga_cap200k_t3corrected.
Historical pages carry a T3 correction notice. Their raw data is retained;
it must not be relabeled as a result on the corrected interval.
