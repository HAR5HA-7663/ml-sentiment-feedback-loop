# Extract tokenizer from existing model and upload to S3

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  EXTRACTING TOKENIZER FROM MODEL" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$MODELS_BUCKET = "ml-sentiment-models-143519759870"
$MODEL_KEY = "training-output/ml-sentiment-training-20251214-032249/output/model.tar.gz"
$TEMP_DIR = "$env:TEMP\sagemaker-model-extract"

Write-Host "Downloading model.tar.gz from S3..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $TEMP_DIR | Out-Null

$modelPath = Join-Path $TEMP_DIR "model.tar.gz"
aws s3 cp "s3://$MODELS_BUCKET/$MODEL_KEY" $modelPath --region us-east-2

if (-not (Test-Path $modelPath)) {
    Write-Host "❌ Failed to download model.tar.gz" -ForegroundColor Red
    exit 1
}

Write-Host "Extracting model.tar.gz..." -ForegroundColor Yellow
$extractDir = Join-Path $TEMP_DIR "extracted"
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

# Extract tar.gz (requires tar or 7zip)
try {
    tar -xzf $modelPath -C $extractDir
} catch {
    Write-Host "❌ Failed to extract. Make sure 'tar' is available or use 7zip" -ForegroundColor Red
    Write-Host "You can manually extract model.tar.gz and find tokenizer.pkl inside" -ForegroundColor Yellow
    exit 1
}

$tokenizerPath = Join-Path $extractDir "tokenizer.pkl"

if (Test-Path $tokenizerPath) {
    Write-Host "✅ Found tokenizer.pkl" -ForegroundColor Green
    Write-Host "Uploading to S3..." -ForegroundColor Yellow
    
    aws s3 cp $tokenizerPath "s3://$MODELS_BUCKET/tokenizers/latest_tokenizer.pkl" --region us-east-2
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Tokenizer uploaded successfully!" -ForegroundColor Green
        Write-Host "   Location: s3://$MODELS_BUCKET/tokenizers/latest_tokenizer.pkl" -ForegroundColor Cyan
    } else {
        Write-Host "❌ Failed to upload tokenizer" -ForegroundColor Red
    }
} else {
    Write-Host "❌ tokenizer.pkl not found in extracted model" -ForegroundColor Red
    Write-Host "   Checked: $tokenizerPath" -ForegroundColor Yellow
}

# Cleanup
Remove-Item -Recurse -Force $TEMP_DIR -ErrorAction SilentlyContinue
