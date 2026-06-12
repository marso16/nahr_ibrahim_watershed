$pipelineConfigurations = @(
    @{ Horizon = 1;  Lookback = 60 }
    @{ Horizon = 3;  Lookback = 60 } 
    @{ Horizon = 14; Lookback = 90 }
)

$seeds = @(69, 42, 2024) 
$hidden = 96
$layers = 4

Write-Host "`n=== Training models ===" -ForegroundColor Cyan

foreach ($config in $pipelineConfigurations) {
    $h = $config.Horizon
    $lookback = $config.Lookback

    foreach ($seed in $seeds) {
        $tag = "tcn_h${h}_s$seed"
        Write-Host "`n--- Horizon $h | Lookback $lookback | Seed $seed | Tag: $tag ---"
        
        $tcnArgs = @(
            "src/models/tcn.py"
            "--horizon", $h
            "--lookback", $lookback
            "--hidden_dim", $hidden
            "--num_layers", $layers
            "--kernel_size", "3"
            "--dropout", "0.30"
            "--lr", "0.0002"
            "--weight_decay", "1e-5"
            "--warmup_epochs", "10"
            "--epochs", "500"
            "--patience", "100"
            "--batch_size", "256"
            "--seed", $seed
            "--run_tag", $tag
        )

        python @tcnArgs
    }
}

python ensemble_tcn.py
Write-Host "`n=== All TCN runs complete ===" -ForegroundColor Green