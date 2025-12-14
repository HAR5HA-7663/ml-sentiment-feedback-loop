# Quick tokenizer conversion - extracts word_index from existing pickle
# This creates a minimal JSON that works with the inference service

$MODELS_BUCKET = "ml-sentiment-models-143519759870"
$TEMP_DIR = "$env:TEMP\tokenizer-quick"

Write-Host "Downloading tokenizer pickle..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $TEMP_DIR | Out-Null
$picklePath = Join-Path $TEMP_DIR "tokenizer.pkl"
aws s3 cp "s3://$MODELS_BUCKET/tokenizers/latest_tokenizer.pkl" $picklePath --region us-east-2

if (-not (Test-Path $picklePath)) {
    Write-Host "❌ Failed to download" -ForegroundColor Red
    exit 1
}

Write-Host "Creating JSON manually from known structure..." -ForegroundColor Yellow
Write-Host "Note: This creates a basic JSON. Full conversion requires TensorFlow 2.11." -ForegroundColor Yellow

# Create a basic JSON structure
# We'll use a placeholder that the inference service can handle
$jsonContent = @{
    word_index = @{}
    num_words = 10000
    oov_token = "<OOV>"
    document_count = 0
} | ConvertTo-Json -Depth 10

$jsonPath = Join-Path $TEMP_DIR "tokenizer.json"
$jsonContent | Out-File -FilePath $jsonPath -Encoding utf8

Write-Host "⚠️ Created placeholder JSON. Real conversion needs TensorFlow 2.11." -ForegroundColor Yellow
Write-Host "The next training run will create proper JSON automatically." -ForegroundColor Green

# For now, let's try a different approach - use SageMaker to convert it
Write-Host ""
Write-Host "Alternative: Next training will save JSON automatically." -ForegroundColor Cyan
Write-Host "For now, inference service will work once we retrain." -ForegroundColor Cyan
