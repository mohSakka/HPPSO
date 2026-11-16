param([ValidateRange(1, 64)][int]$Workers = 2)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$results = 'experiments/results'

function Run-Stage {
    param([string]$Module, [string[]]$Options)
    Write-Output "[$(Get-Date -Format s)] $Module"
    & python -u -m $Module @Options
    if ($LASTEXITCODE -ne 0) { throw "$Module exited with code $LASTEXITCODE" }
}

foreach ($active in @(
    @{ Folder = 'trait_study_coordinatewise_reuse13'; PidFile = 'pipeline.pid'; Signature = 'run_coordinatewise_studies.ps1' },
    @{ Folder = 'openness_ablation_zero_velocity9'; PidFile = 'runner.pid'; Signature = 'experiments.openness_ablation' }
)) {
    $pidFile = Join-Path (Join-Path $results $active.Folder) $active.PidFile
    if (Test-Path -LiteralPath $pidFile) {
        $processId = [int](Get-Content -LiteralPath $pidFile)
        $matchesRun = & python -c 'import psutil,sys; p=int(sys.argv[1]); print(int(psutil.pid_exists(p) and any(sys.argv[2] in a for a in psutil.Process(p).cmdline())))' $processId $active.Signature
        if ($matchesRun -eq '1') {
            Write-Output "Waiting for $($active.Folder) to release its workers"
            Wait-Process -Id $processId -ErrorAction SilentlyContinue
        }
    }
}

$openness = "$results/openness_ablation_zero_velocity9"
$plan = Get-Content -LiteralPath 'experiments/configs/openness_ablation_final9.json' -Raw | ConvertFrom-Json
$saved = @(Get-ChildItem -LiteralPath "$openness/trials" -Recurse -File -Filter 'seed_*.json.gz' -ErrorAction SilentlyContinue).Count
if ($saved -lt ($plan.functions.Count * $plan.seeds.Count * 5)) {
    Run-Stage 'experiments.openness_ablation' @('--config', 'experiments/configs/openness_ablation_final9.json', '--output', $openness, '--workers', $Workers)
} else {
    Run-Stage 'experiments.openness_ablation' @('--output', $openness, '--report')
}
$report = Get-Content -LiteralPath "$openness/report_status.json" -Raw | ConvertFrom-Json
if ($report.saved_trials -ne $report.expected_trials -or $report.failed_trials -ne 0) {
    throw 'The openness report contains incomplete or failed trials'
}
Run-Stage 'experiments.plot_openness_detail' @('--output', $openness)
Run-Stage 'experiments.rerun_velocity_pilots' @('--output', "$results/archived_pilots_pso_zero", '--workers', $Workers)
Write-Output "[$(Get-Date -Format s)] Complete"
