$python_script = "run_pca_comparison.py"
$datasets = @("elec", "sea") # Specify your dataset names here

foreach ($dataset in $datasets) {
    Write-Host "Running analysis for dataset: $dataset" -ForegroundColor Cyan
    
    # Run the python script
    python $python_script --dataset $dataset
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error occurred processing $dataset" -ForegroundColor Red
    } else {
        Write-Host "Completed $dataset" -ForegroundColor Green
    }
    Write-Host "----------------------------------------"
}

Write-Host "All datasets processed." -ForegroundColor Magenta
