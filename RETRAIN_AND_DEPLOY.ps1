# Retrain model with improved hyperparameters and deploy to endpoint
$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"
$REGION = "us-east-2"
$ROLE_ARN = "arn:aws:iam::143519759870:role/ml-sentiment-sagemaker-role"
$MODELS_BUCKET = "ml-sentiment-models-143519759870"
$DATA_BUCKET = "ml-sentiment-data-143519759870"
$ENDPOINT_NAME = "ml-sentiment-endpoint"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RETRAINING WITH IMPROVED HYPERPARAMETERS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create Training Job with improved hyperparameters
Write-Host "Step 1: Creating training job..." -ForegroundColor Yellow
$JOB_NAME = "ml-sentiment-hf-tf-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

$json = @"
{
  "TrainingJobName": "$JOB_NAME",
  "RoleArn": "$ROLE_ARN",
  "AlgorithmSpecification": {
    "TrainingImage": "763104351884.dkr.ecr.$REGION.amazonaws.com/tensorflow-training:2.11-cpu-py39",
    "TrainingInputMode": "File"
  },
  "InputDataConfig": [
    {
      "ChannelName": "train",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "s3://$DATA_BUCKET/",
          "S3DataDistributionType": "FullyReplicated"
        }
      },
      "ContentType": "text/csv"
    }
  ],
  "OutputDataConfig": {
    "S3OutputPath": "s3://$MODELS_BUCKET/training-output/"
  },
  "ResourceConfig": {
    "InstanceType": "ml.m5.large",
    "InstanceCount": 1,
    "VolumeSizeInGB": 30
  },
  "StoppingCondition": {
    "MaxRuntimeInSeconds": 3600
  },
  "HyperParameters": {
    "epochs": "5",
    "batch-size": "16",
    "learning-rate": "1e-5",
    "sagemaker_program": "train_hf_tf.py",
    "sagemaker_submit_directory": "s3://$MODELS_BUCKET/sagemaker-scripts/sourcedir.tar.gz",
    "sagemaker_region": "$REGION"
  },
  "Environment": {
    "MODELS_BUCKET": "$MODELS_BUCKET",
    "S3_MODELS_BUCKET": "$MODELS_BUCKET",
    "S3_DATA_BUCKET": "$DATA_BUCKET"
  }
}
"@

$tempFile = "$env:TEMP\training-$(Get-Date -Format 'yyyyMMddHHmmss').json"
$json | Out-File -FilePath $tempFile -Encoding utf8
aws sagemaker create-training-job --cli-input-json "file://$tempFile" --region $REGION
Remove-Item $tempFile

Write-Host "✅ Training job created: $JOB_NAME" -ForegroundColor Green
Write-Host "   Hyperparameters: epochs=5, batch-size=16, learning-rate=1e-5" -ForegroundColor Cyan
Write-Host ""

# Step 2: Monitor Training
Write-Host "Step 2: Monitoring training (20-30 minutes)..." -ForegroundColor Yellow
$maxWait = 2400
$elapsed = 0
$interval = 60

while ($elapsed -lt $maxWait) {
    Start-Sleep -Seconds $interval
    $elapsed += $interval
    
    $status = aws sagemaker describe-training-job --training-job-name $JOB_NAME --region $REGION --query 'TrainingJobStatus' --output text
    
    Write-Host "[$([math]::Floor($elapsed/60))m] Status: $status" -ForegroundColor $(if ($status -eq "Completed") { "Green" } elseif ($status -eq "Failed") { "Red" } else { "Yellow" })
    
    if ($status -eq "Completed") {
        Write-Host "`n✅ Training completed!" -ForegroundColor Green
        break
    }
    elseif ($status -eq "Failed") {
        $reason = aws sagemaker describe-training-job --training-job-name $JOB_NAME --region $REGION --query 'FailureReason' --output text
        Write-Host "`n❌ FAILED: $reason" -ForegroundColor Red
        Write-Host "`nFix the issue and re-run this script" -ForegroundColor Yellow
        exit 1
    }
}

if ($status -ne "Completed") {
    Write-Host "`n⏱️  Training timeout" -ForegroundColor Yellow
    exit 1
}

# Step 3: Deploy Model
Write-Host "`nStep 3: Deploying model..." -ForegroundColor Yellow

$modelData = aws sagemaker describe-training-job --training-job-name $JOB_NAME --region $REGION --query 'ModelArtifacts.S3ModelArtifacts' --output text
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$MODEL_NAME = "ml-sentiment-hf-model-$timestamp"
$ENDPOINT_CONFIG_NAME = "ml-sentiment-hf-endpoint-config-$timestamp"

# Create model
$modelJson = @"
{
  "ModelName": "$MODEL_NAME",
  "ExecutionRoleArn": "$ROLE_ARN",
  "PrimaryContainer": {
    "Image": "763104351884.dkr.ecr.$REGION.amazonaws.com/tensorflow-training:2.11-cpu-py39",
    "ModelDataUrl": "$modelData",
    "Mode": "SingleModel",
    "Environment": {
      "SAGEMAKER_PROGRAM": "inference_hf_tf.py",
      "SAGEMAKER_SUBMIT_DIRECTORY": "s3://$MODELS_BUCKET/sagemaker-scripts/sourcedir.tar.gz",
      "SAGEMAKER_REGION": "$REGION",
      "SAGEMAKER_CONTAINER_LOG_LEVEL": "20"
    }
  }
}
"@

$modelFile = "$env:TEMP\model-$timestamp.json"
$modelJson | Out-File -FilePath $modelFile -Encoding utf8
aws sagemaker create-model --cli-input-json "file://$modelFile" --region $REGION
Remove-Item $modelFile

# Create endpoint config
$configJson = @"
{
  "EndpointConfigName": "$ENDPOINT_CONFIG_NAME",
  "ProductionVariants": [
    {
      "VariantName": "AllTraffic",
      "ModelName": "$MODEL_NAME",
      "InstanceType": "ml.t2.medium",
      "InitialInstanceCount": 1,
      "InitialVariantWeight": 1.0
    }
  ]
}
"@

$configFile = "$env:TEMP\config-$timestamp.json"
$configJson | Out-File -FilePath $configFile -Encoding utf8
aws sagemaker create-endpoint-config --cli-input-json "file://$configFile" --region $REGION
Remove-Item $configFile

# Update endpoint
$endpointExists = $false
try {
    $existingEndpoint = aws sagemaker describe-endpoint --endpoint-name $ENDPOINT_NAME --region $REGION 2>$null | ConvertFrom-Json
    if ($existingEndpoint) {
        $endpointExists = $true
    }
} catch {
    $endpointExists = $false
}

if ($endpointExists) {
    Write-Host "Updating existing endpoint..." -ForegroundColor Yellow
    aws sagemaker update-endpoint --endpoint-name $ENDPOINT_NAME --endpoint-config-name $ENDPOINT_CONFIG_NAME --region $REGION
} else {
    Write-Host "Creating new endpoint..." -ForegroundColor Yellow
    $endpointJson = @"
{
  "EndpointName": "$ENDPOINT_NAME",
  "EndpointConfigName": "$ENDPOINT_CONFIG_NAME"
}
"@
    $endpointFile = "$env:TEMP\endpoint-$timestamp.json"
    $endpointJson | Out-File -FilePath $endpointFile -Encoding utf8
    aws sagemaker create-endpoint --cli-input-json "file://$endpointFile" --region $REGION
    Remove-Item $endpointFile
}

Write-Host "✅ Deployment started!" -ForegroundColor Green
Write-Host ""

# Step 4: Wait for Endpoint
Write-Host "Step 4: Waiting for endpoint (3-5 minutes)..." -ForegroundColor Yellow
$maxWait = 600
$elapsed = 0
$interval = 20

while ($elapsed -lt $maxWait) {
    Start-Sleep -Seconds $interval
    $elapsed += $interval
    
    $epStatus = aws sagemaker describe-endpoint --endpoint-name $ENDPOINT_NAME --region $REGION --query 'EndpointStatus' --output text
    
    Write-Host "[$([math]::Floor($elapsed/60))m] Endpoint: $epStatus" -ForegroundColor $(if ($epStatus -eq "InService") { "Green" } else { "Yellow" })
    
    if ($epStatus -eq "InService") {
        Write-Host "`n✅ Endpoint ready!" -ForegroundColor Green
        break
    }
    elseif ($epStatus -eq "Failed") {
        Write-Host "`n❌ Endpoint failed!" -ForegroundColor Red
        exit 1
    }
}

if ($epStatus -ne "InService") {
    Write-Host "`n⏱️  Endpoint timeout" -ForegroundColor Yellow
    exit 1
}

Start-Sleep -Seconds 10

# Step 5: Test for Overfitting/Underfitting
Write-Host "`nStep 5: Testing for overfitting/underfitting..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

$testCases = @(
    @{text="nice size, very clear but randomly shuts off. cant remove unwanted apps"; expected="negative"},
    @{text="This tablet is on the smaller side but works for me."; expected="neutral"},
    @{text="Too difficult to setup not compatible"; expected="negative"},
    @{text="I absolutely LOVE my kindle"; expected="positive"},
    @{text="Ended up returning"; expected="negative"}
)

$results = @()
$correct = 0
$allPositive = $true
$allNeutral = $true
$predictions = @{}

foreach ($tc in $testCases) {
    try {
        $body = @{ text = $tc.text } | ConvertTo-Json
        $response = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 15
        
        $isCorrect = ($response.label -eq $tc.expected)
        if ($isCorrect) { $correct++ }
        if ($response.label -ne "positive") { $allPositive = $false }
        if ($response.label -ne "neutral") { $allNeutral = $false }
        
        if (-not $predictions.ContainsKey($response.label)) {
            $predictions[$response.label] = 0
        }
        $predictions[$response.label]++
        
        $color = if ($isCorrect) { "Green" } else { "Red" }
        Write-Host "Expected: $($tc.expected.PadRight(8)) | Got: $($response.label.PadRight(8)) | Conf: $([math]::Round($response.confidence, 3))" -ForegroundColor $color
    }
    catch {
        Write-Host "Error: $_" -ForegroundColor Red
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
$accuracy = [math]::Round($correct/$testCases.Count*100, 1)
Write-Host "Accuracy: $correct/$($testCases.Count) = ${accuracy}%" -ForegroundColor $(if ($accuracy -ge 70) { "Green" } elseif ($accuracy -ge 50) { "Yellow" } else { "Red" })

Write-Host "`nPrediction Distribution:" -ForegroundColor Cyan
foreach ($pred in $predictions.GetEnumerator() | Sort-Object -Property Value -Descending) {
    Write-Host "  $($pred.Key): $($pred.Value)" -ForegroundColor White
}

# Final Result
if ($allPositive) {
    Write-Host "`n❌ STILL OVERFITTING: Only positive predictions" -ForegroundColor Red
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "1. Increase regularization (dropout, L2)" -ForegroundColor White
    Write-Host "2. Reduce learning rate further" -ForegroundColor White
    Write-Host "3. Try different model architecture" -ForegroundColor White
    exit 1
}
elseif ($allNeutral) {
    Write-Host "`n❌ STILL UNDERFITTING: Only neutral predictions" -ForegroundColor Red
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "1. Increase epochs (5 → 10)" -ForegroundColor White
    Write-Host "2. Increase learning rate (1e-5 → 2e-5)" -ForegroundColor White
    Write-Host "3. Check training data quality" -ForegroundColor White
    exit 1
}
else {
    Write-Host "`n✅ MODEL WORKING! Predicts multiple classes!" -ForegroundColor Green
    Write-Host "`nReady to commit and deploy via GitHub Actions!" -ForegroundColor Cyan
    Write-Host "`nRun: git add -A; git commit -m 'Fix overfitting with improved hyperparameters'; git push origin main" -ForegroundColor White
    exit 0
}
