param(
    [ValidateSet("status", "watch", "pause", "resume", "pids")]
    [string]$Action = "status",
    [string]$Profile = "fast_paper_v1"
)

$ErrorActionPreference = "Stop"

# repo root: .../paper/scripts -> .../
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

$Profiles = @{
    "clean_v4" = @{
        python           = "C:\Python312\python.exe"
        from_phase       = "C"
        to_phase         = "F"
        include_expected = $true
        suite_dir        = "runs/suite_clean_v4"
        sdft_data        = "data/sdft/from_ttt_train_clean_v4.jsonl"
        sdft_out         = "runs/sdft_from_ttt_steps500_clean_v4"
        fig_dir          = "paper/figs_clean_v4"
        orchestrator_dir = "runs/orchestrator_clean_v4"
        logs_dir         = "runs/orchestrator_clean_v4/logs"
        state_path       = "runs/orchestrator_clean_v4/state.json"
    }
    "fast_paper_v1" = @{
        python           = "C:\Python312\python.exe"
        from_phase       = "C"
        to_phase         = "F"
        include_expected = $true
        suite_dir        = "runs/suite_fast_v1"
        sdft_data        = "data/sdft/from_ttt_train_fast_v1.jsonl"
        sdft_out         = "runs/sdft_from_ttt_steps300_fast_v1"
        fig_dir          = "paper/figs_fast_v1"
        orchestrator_dir = "runs/orchestrator_fast_v1"
        logs_dir         = "runs/orchestrator_fast_v1/logs"
        state_path       = "runs/orchestrator_fast_v1/state.json"
        ttt_steps        = 12
        rollouts         = 8
        max_new_tokens   = 16
        sdft_steps       = 300
        min_reward       = 0.95
    }
}

if (-not $Profiles.ContainsKey($Profile)) {
    throw "Unknown profile '$Profile'. Available: $($Profiles.Keys -join ', ')"
}

$Cfg = $Profiles[$Profile]
$OrchestratorNeedle = "--orchestrator-dir $($Cfg.orchestrator_dir)"

function Get-OrchestratorProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object {
            $_.CommandLine -like "*paper/scripts/run_phase.py*" -and
            $_.CommandLine -like "*$OrchestratorNeedle*"
        } |
        Select-Object -First 1
}

function Show-Pids {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object {
            $_.CommandLine -like "*$($Cfg.orchestrator_dir)*" -or
            $_.CommandLine -like "*$($Cfg.suite_dir)*"
        } |
        Select-Object ProcessId, ParentProcessId, CommandLine |
        Format-List
}

function Show-Status {
    Write-Host ""
    Write-Host "[suite]"
    & $Cfg.python "paper/scripts/run_status.py" --root $Cfg.suite_dir
    Write-Host ""
    Write-Host "[orchestrator]"
    & $Cfg.python "paper/scripts/run_status.py" --root $Cfg.orchestrator_dir
}

function Watch-Status {
    Write-Host ""
    Write-Host "[watching suite: Ctrl+C to stop]"
    & $Cfg.python "paper/scripts/run_status.py" --root $Cfg.suite_dir --watch 15
}

function Pause-Run {
    $proc = Get-OrchestratorProcess
    if (-not $proc) {
        Write-Host "No orchestrator process found for profile '$Profile'."
        return
    }
    & taskkill /PID $proc.ProcessId /T /F | Out-Host
    Write-Host "Paused profile '$Profile'."
}

function Resume-Run {
    $existing = Get-OrchestratorProcess
    if ($existing) {
        Write-Host "Run is already active (PID $($existing.ProcessId))."
        return
    }

    $args = @(
        "paper/scripts/run_phase.py",
        "--from-phase", $Cfg.from_phase,
        "--to-phase", $Cfg.to_phase,
        "--suite-dir", $Cfg.suite_dir,
        "--sdft-data", $Cfg.sdft_data,
        "--sdft-out", $Cfg.sdft_out,
        "--fig-dir", $Cfg.fig_dir,
        "--orchestrator-dir", $Cfg.orchestrator_dir,
        "--logs-dir", $Cfg.logs_dir,
        "--state-path", $Cfg.state_path
    )
    if ($null -ne $Cfg.ttt_steps) {
        $args += @("--ttt-steps", "$($Cfg.ttt_steps)")
    }
    if ($null -ne $Cfg.rollouts) {
        $args += @("--rollouts", "$($Cfg.rollouts)")
    }
    if ($null -ne $Cfg.max_new_tokens) {
        $args += @("--max-new-tokens", "$($Cfg.max_new_tokens)")
    }
    if ($null -ne $Cfg.sdft_steps) {
        $args += @("--sdft-steps", "$($Cfg.sdft_steps)")
    }
    if ($null -ne $Cfg.min_reward) {
        $args += @("--min-reward", "$($Cfg.min_reward)")
    }
    if ($Cfg.include_expected) {
        $args += "--include-expected-ablation"
    }

    $proc = Start-Process -FilePath $Cfg.python -ArgumentList $args -WorkingDirectory $RepoRoot -PassThru
    Write-Host "Resumed profile '$Profile' (PID $($proc.Id))."
}

switch ($Action) {
    "status" { Show-Status; break }
    "watch"  { Watch-Status; break }
    "pause"  { Pause-Run; break }
    "resume" { Resume-Run; break }
    "pids"   { Show-Pids; break }
    default  { throw "Unsupported action: $Action" }
}
