# Convert tokenizer using Docker (avoids local TensorFlow version issues)

$MODELS_BUCKET = "ml-sentiment-models-143519759870"
$TEMP_DIR = "$env:TEMP\model-extract-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

Write-Host "Creating temp directory..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $TEMP_DIR | Out-Null

Write-Host "Downloading model.tar.gz..." -ForegroundColor Yellow
$modelPath = Join-Path $TEMP_DIR "model.tar.gz"
aws s3 cp "s3://$MODELS_BUCKET/training-output/ml-sentiment-training-20251214-032249/output/model.tar.gz" $modelPath --region us-east-2

Write-Host "Extracting model.tar.gz..." -ForegroundColor Yellow
tar -xzf $modelPath -C $TEMP_DIR

$tokenizerPath = Join-Path $TEMP_DIR "tokenizer.pkl"
if (-not (Test-Path $tokenizerPath)) {
    Write-Host "❌ Tokenizer not found" -ForegroundColor Red
    exit 1
}

Write-Host "Converting using Docker Python 3.9 + TensorFlow 2.11..." -ForegroundColor Yellow

$convertScript = @"
import pickle
import json
import os

tokenizer = pickle.load(open('/data/tokenizer.pkl', 'rb'))
data = {
    'word_index': tokenizer.word_index,
    'num_words': getattr(tokenizer, 'num_words', 10000),
    'oov_token': getattr(tokenizer, 'oov_token', '<OOV>'),
    'document_count': getattr(tokenizer, 'document_count', 0)
}

with open('/data/tokenizer.json', 'w') as f:
    json.dump(data, f)

word_count = len(data['word_index'])
print('Success! Word index size: ' + str(word_count))
"@

$scriptPath = Join-Path $TEMP_DIR "convert.py"
$convertScript | Out-File -FilePath $scriptPath -Encoding utf8

# Convert Windows path to Docker volume format
$dockerPath = $TEMP_DIR -replace '\\', '/' -replace 'C:', '/c' -replace ':', ''

docker run --rm -v "${TEMP_DIR}:/data" python:3.9-slim bash -c "pip install -q 'numpy<2.0' tensorflow==2.11.0 && python /data/convert.py"

$jsonPath = Join-Path $TEMP_DIR "tokenizer.json"
if (Test-Path $jsonPath) {
    Write-Host "Uploading JSON to S3..." -ForegroundColor Yellow
    aws s3 cp $jsonPath "s3://$MODELS_BUCKET/tokenizers/latest_tokenizer.json" --region us-east-2
    Write-Host "✅ Tokenizer JSON uploaded!" -ForegroundColor Green
} else {
    Write-Host "❌ JSON conversion failed" -ForegroundColor Red
    exit 1
}

Remove-Item -Recurse -Force $TEMP_DIR -ErrorAction SilentlyContinue
