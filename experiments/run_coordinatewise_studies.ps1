param([ValidateRange(1, 64)][int]$Workers = 2)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$config = 'experiments/configs/trait_study_coordinatewise13.json'
$traits = 'experiments/results/trait_study_coordinatewise_reuse13'
$diversity = 'experiments/results/diversity_coordinatewise13'

function Run-Study {
    param([string]$Module, [string[]]$Options)
    Write-Output "[$(Get-Date -Format s)] $Module"
    & python -u -m $Module @Options
    if ($LASTEXITCODE -ne 0) {
        throw "$Module exited with code $LASTEXITCODE"
    }
}

Run-Study 'experiments.trait_study' @('--config', $config, '--output', $traits, '--workers', $Workers)
Run-Study 'experiments.trait_report' @('--output', $traits, '--sweep-curves')
Run-Study 'experiments.audit_trait_study' @('--output', $traits)
Run-Study 'experiments.merge_coordinatewise_diversity' @('--ablation', $traits, '--original', 'experiments/results/diversity_zero_velocity13', '--output', $diversity)
Run-Study 'experiments.plot_diversity_early' @('--output', $diversity)
Write-Output "[$(Get-Date -Format s)] Complete"
