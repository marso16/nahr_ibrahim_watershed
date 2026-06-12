$ROOT = "C:/Users/marck/Downloads/nahr_ibrahim_watershed"
$SCRIPT = "$ROOT/src/models/gr4j_tcn.py"

$pipelineConfigurations = @(
    @{ Horizon = 1;  Lookback = 60 }
    @{ Horizon = 3;  Lookback = 60 }
    @{ Horizon = 14; Lookback = 90 }
)

$seeds = @(42, 69, 2024)
$hidden = 96
$layers = 4

$total = $pipelineConfigurations.Length * $seeds.Length
$current = 0
$startTime = Get-Date

Write-Host "`n=== GR4J-TCN Path 2: fair head-to-head with plain TCN ===" -ForegroundColor Cyan
Write-Host "Total runs: $total" -ForegroundColor Cyan
Write-Host "Started:    $startTime" -ForegroundColor Cyan

foreach ($config in $pipelineConfigurations) {
    $h = $config.Horizon
    $lookback = $config.Lookback

    foreach ($seed in $seeds) {
        $current++
        $tag = "gr4j_tcn_h${h}_s$seed"

        Write-Host "`n============================================================" -ForegroundColor Cyan
        Write-Host "  [$current/$total] Horizon=$h | Lookback=$lookback | Seed=$seed" -ForegroundColor Cyan
        Write-Host "  Tag: $tag" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan

        python $SCRIPT `
            --horizon $h `
            --lookback $lookback `
            --seq_suffix "_hybrid" `
            --precip_idx 0 `
            --pet_idx 27 `
            --hidden_dim $hidden `
            --num_layers $layers `
            --kernel_size 3 `
            --dropout 0.30 `
            --lr 0.0002 `
            --weight_decay 1e-5 `
            --warmup_epochs 10 `
            --epochs 500 `
            --patience 100 `
            --batch_size 256 `
            --loss nse `
            --ema_decay 0.999 `
            --seed $seed `
            --run_tag $tag

        if ($LASTEXITCODE -eq 0) {
            Write-Host "  DONE: $tag" -ForegroundColor Green
        } else {
            Write-Host "  FAILED (exit $LASTEXITCODE): $tag" -ForegroundColor Red
            Write-Host "  Stopping early so you can inspect the failure." -ForegroundColor Yellow
            break
        }
    }
    # Break the outer loop too if a run failed
    if ($LASTEXITCODE -ne 0) { break }
}

$endTime = Get-Date
$elapsed = $endTime - $startTime

Write-Host "`n=== GR4J-TCN Path 2 complete ===" -ForegroundColor Green
Write-Host "Finished:  $endTime" -ForegroundColor Green
Write-Host "Elapsed:   $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host ""
Write-Host "Next step: collect metrics for the head-to-head:" -ForegroundColor Cyan
Write-Host "  python src/utility/collect_metrics.py" -ForegroundColor White