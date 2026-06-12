$ROOT = "C:/Users/marck/Downloads/nahr_ibrahim_watershed"
$SCRIPT = "$ROOT/src/models/gr4j_tcn.py"

$JOBS = @(
    @{ h = 1;  lb = 60; bs = 512; tag = "gr4j_tcn_h1_s69" }
    @{ h = 3;  lb = 60; bs = 512; tag = "gr4j_tcn_h3_s69" }
    @{ h = 14; lb = 90; bs = 256; tag = "gr4j_tcn_h14_s69" }
)

$total = $JOBS.Length
$current = 0

foreach ($job in $JOBS) {
    $current++
    $h   = $job.h
    $lb  = $job.lb
    $bs  = $job.bs
    $tag = $job.tag

    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "  [$current/$total] Training GR4J-TCN: horizon=$h, lookback=$lb" -ForegroundColor Cyan
    Write-Host "  Batch: $bs | Tag: $tag" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    python $SCRIPT `
        --horizon $h `
        --lookback $lb `
        --seq_suffix "_hybrid" `
        --precip_idx 0 `
        --pet_idx 27 `
        --hidden_dim 96 `
        --num_layers 4 `
        --kernel_size 3 `
        --dropout 0.30 `
        --lr 0.0002 `
        --weight_decay 1e-5 `
        --warmup_epochs 10 `
        --epochs 500 `
        --patience 100 `
        --batch_size $bs `
        --loss huber `
        --seed 69 `
        --run_tag $tag

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  DONE: $tag" -ForegroundColor Green
    } else {
        Write-Host "  FAILED (exit $LASTEXITCODE): $tag" -ForegroundColor Red
        break
    }
}

Write-Host "`nAll GR4J-TCN training runs complete." -ForegroundColor Green