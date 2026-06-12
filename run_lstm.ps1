$pipelineConfigurations = @(
    @{ Horizon = 1;  Lookback = 60 }
    @{ Horizon = 3;  Lookback = 60 }
    @{ Horizon = 14; Lookback = 90 }
)

$seeds = @(42, 69, 2024)

Write-Host "`n=== Training LSTM models ===" -ForegroundColor Cyan

foreach ($config in $pipelineConfigurations) {
    $h = $config.Horizon
    $lookback = $config.Lookback

    foreach ($seed in $seeds) {
        $tag = "lstm_h${h}_s$seed"
        Write-Host "`n--- Horizon $h | Lookback $lookback | Seed $seed | Tag: $tag ---"

        $lstmArgs = @(
            "src/models/lstm.py"
            "--horizon", $h
            "--lookback", $lookback
            "--units_1", "128"
            "--units_2", "96"
            "--dropout", "0.30"
            "--recurrent_dropout", "0.20"
            "--lr", "1.0e-04"
            "--weight_decay", "1e-5"
            "--warmup_epochs", "10"
            "--epochs", "500"
            "--patience", "100"
            "--batch_size", "256"
            "--loss", "huber"
            "--seed", $seed
            "--run_tag", $tag
        )

        python @lstmArgs
    }
}

python ensemble_lstm.py
Write-Host "`n=== All LSTM runs complete ===" -ForegroundColor Green