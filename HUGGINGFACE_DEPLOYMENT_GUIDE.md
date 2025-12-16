# Hugging Face Model Deployment Guide - Complete Workflow

## 🎯 Goal

Deploy TensorFlow + Hugging Face model (TFDistilBERT) and test until overfitting is fixed.

## ✅ What's Already Done

- ✅ `train_hf_tf.py` - Fixed save operations (saves to /tmp first, then copies)
- ✅ `inference_hf_tf.py` - Loads tokenizer from pickle
- ✅ `model-init-service` - Updated to use `train_hf_tf.py` and `inference_hf_tf.py`
- ✅ `sourcedir.tar.gz` - Packaged and uploaded to S3

## 📋 Step-by-Step Deployment

### Step 1: Verify Files in S3

```powershell
aws s3 ls s3://ml-sentiment-models-143519759870/sagemaker-scripts/ --region us-east-2
# Should see: sourcedir.tar.gz (latest version)
```

### Step 2: Create Training Job via CLI

```powershell
$JOB_NAME = "ml-sentiment-hf-tf-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$json = @"
{
  "TrainingJobName": "$JOB_NAME",
  "RoleArn": "arn:aws:iam::143519759870:role/ml-sentiment-sagemaker-role",
  "AlgorithmSpecification": {
    "TrainingImage": "763104351884.dkr.ecr.us-east-2.amazonaws.com/tensorflow-training:2.11-cpu-py39",
    "TrainingInputMode": "File"
  },
  "InputDataConfig": [
    {
      "ChannelName": "train",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "s3://ml-sentiment-data-143519759870/",
          "S3DataDistributionType": "FullyReplicated"
        }
      },
      "ContentType": "text/csv"
    }
  ],
  "OutputDataConfig": {
    "S3OutputPath": "s3://ml-sentiment-models-143519759870/training-output/"
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
    "epochs": "3",
    "batch-size": "16",
    "learning-rate": "2e-5",
    "sagemaker_program": "train_hf_tf.py",
    "sagemaker_submit_directory": "s3://ml-sentiment-models-143519759870/sagemaker-scripts/sourcedir.tar.gz",
    "sagemaker_region": "us-east-2"
  },
  "Environment": {
    "MODELS_BUCKET": "ml-sentiment-models-143519759870",
    "S3_MODELS_BUCKET": "ml-sentiment-models-143519759870",
    "S3_DATA_BUCKET": "ml-sentiment-data-143519759870"
  }
}
"@

$tempFile = "$env:TEMP\training-$(Get-Date -Format 'yyyyMMddHHmmss').json"
$json | Out-File -FilePath $tempFile -Encoding utf8
aws sagemaker create-training-job --cli-input-json "file://$tempFile" --region us-east-2
Remove-Item $tempFile

Write-Host "✅ Training job created: $JOB_NAME" -ForegroundColor Green
```

### Step 3: Monitor Training (Wait 20-30 minutes)

```powershell
$JOB_NAME = "ml-sentiment-hf-tf-YYYYMMDD-HHMMSS"  # Use actual job name from Step 2

$maxWait = 2400  # 40 minutes
$elapsed = 0
$interval = 60  # Check every minute

while ($elapsed -lt $maxWait) {
    Start-Sleep -Seconds $interval
    $elapsed += $interval

    $status = aws sagemaker describe-training-job --training-job-name $JOB_NAME --region us-east-2 --query 'TrainingJobStatus' --output text

    Write-Host "[$([math]::Floor($elapsed/60))m] Status: $status" -ForegroundColor $(if ($status -eq "Completed") { "Green" } elseif ($status -eq "Failed") { "Red" } else { "Yellow" })

    if ($status -eq "Completed") {
        Write-Host "`n✅ Training completed!" -ForegroundColor Green
        break
    }
    elseif ($status -eq "Failed") {
        $reason = aws sagemaker describe-training-job --training-job-name $JOB_NAME --region us-east-2 --query 'FailureReason' --output text
        Write-Host "`n❌ FAILED: $reason" -ForegroundColor Red
        break
    }
}
```

### Step 4: Deploy Model to Endpoint

```powershell
$JOB_NAME = "ml-sentiment-hf-tf-YYYYMMDD-HHMMSS"  # Use completed job name

# Get model artifacts
$modelData = aws sagemaker describe-training-job --training-job-name $JOB_NAME --region us-east-2 --query 'ModelArtifacts.S3ModelArtifacts' --output text
Write-Host "Model artifacts: $modelData" -ForegroundColor Cyan

# Create model
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$MODEL_NAME = "ml-sentiment-hf-model-$timestamp"
$ENDPOINT_CONFIG_NAME = "ml-sentiment-hf-endpoint-config-$timestamp"
$ENDPOINT_NAME = "ml-sentiment-endpoint"
$ROLE_ARN = "arn:aws:iam::143519759870:role/ml-sentiment-sagemaker-role"
$REGION = "us-east-2"
$MODELS_BUCKET = "ml-sentiment-models-143519759870"

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

$modelFile = "$env:TEMP\model-hf-$timestamp.json"
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

$configFile = "$env:TEMP\config-hf-$timestamp.json"
$configJson | Out-File -FilePath $configFile -Encoding utf8
aws sagemaker create-endpoint-config --cli-input-json "file://$configFile" --region $REGION
Remove-Item $configFile

# Update endpoint
aws sagemaker update-endpoint --endpoint-name $ENDPOINT_NAME --endpoint-config-name $ENDPOINT_CONFIG_NAME --region $REGION

Write-Host "✅ Deployment started!" -ForegroundColor Green
```

### Step 5: Wait for Endpoint (3-5 minutes)

```powershell
$ENDPOINT_NAME = "ml-sentiment-endpoint"
$REGION = "us-east-2"

$maxWait = 600  # 10 minutes
$elapsed = 0
$interval = 20

while ($elapsed -lt $maxWait) {
    Start-Sleep -Seconds $interval
    $elapsed += $interval

    $status = aws sagemaker describe-endpoint --endpoint-name $ENDPOINT_NAME --region $REGION --query 'EndpointStatus' --output text

    Write-Host "[$([math]::Floor($elapsed/60))m] Status: $status" -ForegroundColor $(if ($status -eq "InService") { "Green" } else { "Yellow" })

    if ($status -eq "InService") {
        Write-Host "`n✅ Endpoint ready!" -ForegroundColor Green
        break
    }
    elseif ($status -eq "Failed") {
        Write-Host "`n❌ Endpoint failed!" -ForegroundColor Red
        break
    }
}
```

### Step 6: Test for Overfitting

```powershell
$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "TESTING FOR OVERFITTING" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

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
$predictions = @{}

foreach ($tc in $testCases) {
    try {
        $body = @{ text = $tc.text } | ConvertTo-Json
        $response = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 15

        $isCorrect = ($response.label -eq $tc.expected)
        if ($isCorrect) { $correct++ }
        if ($response.label -ne "positive") { $allPositive = $false }

        if (-not $predictions.ContainsKey($response.label)) {
            $predictions[$response.label] = 0
        }
        $predictions[$response.label]++

        $color = if ($isCorrect) { "Green" } else { "Red" }
        Write-Host "Expected: $($tc.expected.PadRight(8)) | Got: $($response.label.PadRight(8)) | Conf: $([math]::Round($response.confidence, 3))" -ForegroundColor $color

        $results += [PSCustomObject]@{
            Expected=$tc.expected
            Predicted=$response.label
            Confidence=[math]::Round($response.confidence, 3)
            Correct=if($isCorrect){"✅"}else{"❌"}
        }
    }
    catch {
        Write-Host "Error: $_" -ForegroundColor Red
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
$accuracy = [math]::Round($correct/$testCases.Count*100, 1)
Write-Host "Results: $correct/$($testCases.Count) = ${accuracy}%" -ForegroundColor $(if ($accuracy -ge 70) { "Green" } elseif ($accuracy -ge 50) { "Yellow" } else { "Red" })

Write-Host "`nPrediction Distribution:" -ForegroundColor Cyan
foreach ($pred in $predictions.GetEnumerator() | Sort-Object -Property Value -Descending) {
    Write-Host "  $($pred.Key): $($pred.Value)" -ForegroundColor White
}

if ($allPositive) {
    Write-Host "`n❌ STILL OVERFITTING: Only positive predictions" -ForegroundColor Red
    Write-Host "   Need to adjust training parameters or try different approach" -ForegroundColor Yellow
    exit 1  # Signal failure - need to retrain
}
else {
    Write-Host "`n✅ OVERFITTING FIXED! Model predicts multiple classes!" -ForegroundColor Green
    Write-Host "   Ready for GitHub Actions deployment!" -ForegroundColor Cyan
    exit 0  # Success
}
```

## 🔄 Iteration Loop

If overfitting persists:

1. **Adjust training parameters:**

   - Increase epochs (3 → 5)
   - Adjust learning rate (2e-5 → 1e-5 or 3e-5)
   - Change batch size (16 → 8 or 32)

2. **Modify data balancing:**

   - Try different undersampling ratio (currently 1:1)
   - Or try oversampling minority classes

3. **Change model:**

   - Try `bert-base-uncased` instead of `distilbert-base-uncased`
   - Or adjust model architecture

4. **Re-run Steps 2-6** until overfitting is fixed

## ✅ Success Criteria

- Model predicts **negative** and **neutral** classes (not just positive)
- Test accuracy > 60% on diverse examples
- Prediction distribution shows all 3 classes

## 📝 Notes

- Training takes 20-30 minutes (CPU instance)
- Endpoint deployment takes 3-5 minutes
- Use AWS CLI for all operations (no git push until it works)
- Test thoroughly before committing to GitHub
