# Convert existing tokenizer pickle to JSON format

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  CONVERTING TOKENIZER TO JSON" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$MODELS_BUCKET = "ml-sentiment-models-143519759870"
$TEMP_DIR = "$env:TEMP\tokenizer-convert"

Write-Host "Step 1: Downloading tokenizer pickle..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $TEMP_DIR | Out-Null

$picklePath = Join-Path $TEMP_DIR "tokenizer.pkl"
aws s3 cp "s3://$MODELS_BUCKET/tokenizers/latest_tokenizer.pkl" $picklePath --region us-east-2

if (-not (Test-Path $picklePath)) {
    Write-Host "❌ Failed to download tokenizer" -ForegroundColor Red
    exit 1
}

Write-Host "Step 2: Converting to JSON..." -ForegroundColor Yellow
Write-Host "   (Using Python to load pickle and extract data)" -ForegroundColor White

$pythonScript = @"
import pickle
import json
import sys

# Load tokenizer
with open(r'$picklePath', 'rb') as f:
    tokenizer = pickle.load(f)

# Extract data
tokenizer_data = {
    'word_index': tokenizer.word_index,
    'num_words': getattr(tokenizer, 'num_words', 10000),
    'oov_token': getattr(tokenizer, 'oov_token', '<OOV>'),
    'document_count': getattr(tokenizer, 'document_count', 0)
}

# Save as JSON
json_path = r'$TEMP_DIR\tokenizer.json'
with open(json_path, 'w') as f:
    json.dump(tokenizer_data, f)

print(f'Successfully converted to {json_path}')
print(f'Word index size: {len(tokenizer_data[\"word_index\"])}')
"@

$scriptPath = Join-Path $TEMP_DIR "convert.py"
$pythonScript | Out-File -FilePath $scriptPath -Encoding utf8

python $scriptPath
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Conversion failed!" -ForegroundColor Red
    Write-Host "Make sure Python with TensorFlow 2.11 is available" -ForegroundColor Yellow
    exit 1
}

$jsonPath = Join-Path $TEMP_DIR "tokenizer.json"
if (-not (Test-Path $jsonPath)) {
    Write-Host "❌ JSON file not created" -ForegroundColor Red
    exit 1
}

Write-Host "Step 3: Uploading JSON to S3..." -ForegroundColor Yellow
aws s3 cp $jsonPath "s3://$MODELS_BUCKET/tokenizers/latest_tokenizer.json" --region us-east-2

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Tokenizer JSON uploaded successfully!" -ForegroundColor Green
    Write-Host "   Location: s3://$MODELS_BUCKET/tokenizers/latest_tokenizer.json" -ForegroundColor Cyan
} else {
    Write-Host "❌ Upload failed" -ForegroundColor Red
    exit 1
}

# Cleanup
Remove-Item -Recurse -Force $TEMP_DIR -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  ✅ CONVERSION COMPLETE!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
