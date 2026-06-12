$ROOT = "C:/Users/marck/Downloads/nahr_ibrahim_watershed"
$CHECKPOINT_DIR = "$ROOT/models/checkpoints"
$UTILITY = "$ROOT/src/utility/run_projections.py"

$MODEL_MAP = @{
    "gr4j_tcn"  = @{ file = "src/models/gr4j_tcn.py";  class = "HybridGR4J_TCN";  is_hybrid = $true }
    "tcn"       = @{ file = "src/models/tcn.py";       class = "WatershedTCN";    is_hybrid = $false }
    "lstm"      = @{ file = "src/models/lstm.py";      class = "WatershedLSTM";   is_hybrid = $false }
}

function Get-Lookback($h) {
    if ($h -eq 14) { return 90 }
    return 60   # h=1, h=3, and any other
}

function Resolve-Scaler($family, $horizon, $lookback, $is_hybrid) {
    $candidates = @()
    if ($is_hybrid) {
        $candidates += "$ROOT/data/splits/scaler_params_h${horizon}_lb${lookback}_hybrid.csv"
    }
    $candidates += "$ROOT/data/splits/scaler_params_h${horizon}_lb${lookback}.csv"
    $candidates += "$ROOT/data/splits/scaler_params.csv"
    
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

$checkpoints = Get-ChildItem -Path $CHECKPOINT_DIR -Filter "*_best_*.pt" | Sort-Object Name

if ($checkpoints.Count -eq 0) {
    Write-Host "No checkpoints found in $CHECKPOINT_DIR" -ForegroundColor Red
    exit 1
}

Write-Host "Found $($checkpoints.Count) checkpoint(s)" -ForegroundColor Green

foreach ($ckpt in $checkpoints) {
    $stem = $ckpt.BaseName

    if ($stem -notmatch "^(.+?)_best_(.+)$") {
        Write-Host "  SKIP: unrecognized naming pattern: $($ckpt.Name)" -ForegroundColor Yellow
        continue
    }
    $family = $matches[1]
    $tag    = $matches[2]

    if (-not $MODEL_MAP.ContainsKey($family)) {
        Write-Host "  SKIP: unknown family '$family' for $stem. Add to MODEL_MAP." -ForegroundColor Yellow
        continue
    }
    $m = $MODEL_MAP[$family]

    if ($tag -match "_h(\d+)_") {
        $horizon = [int]$matches[1]
    } else {
        $horizon = 1
        Write-Host "  WARNING: no horizon found in $tag, assuming h=1" -ForegroundColor Yellow
    }

    $lookback = Get-Lookback $horizon
    $scaler = Resolve-Scaler $family $horizon $lookback $m.is_hybrid

    if ($scaler -eq $null) {
        Write-Host "  SKIP: no scaler found for $family h=$horizon lb=$lookback." -ForegroundColor Red
        continue
    }

    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "  Model : $family" -ForegroundColor Cyan
    Write-Host "  Tag   : $tag" -ForegroundColor Cyan
    Write-Host "  Horizon: $horizon  |  Lookback: $lookback" -ForegroundColor Cyan
    Write-Host "  Scaler: $scaler" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    python $UTILITY `
        --checkpoint "$($ckpt.FullName)" `
        --model_file "$ROOT/$($m.file)" `
        --model_class $m.class `
        --scaler_csv $scaler `
        --lookback $lookback `
        --horizon $horizon `
        --batch_size 512

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED (exit $LASTEXITCODE): $stem" -ForegroundColor Red
    } else {
        Write-Host "  DONE: $stem" -ForegroundColor Green
    }
}

Write-Host "`nAll projections complete. Output in $ROOT/data/projections/" -ForegroundColor Green