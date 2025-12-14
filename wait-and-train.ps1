# Complete Automated Training Script
# Waits for deployment, then triggers training automatically

$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  AUTOMATED SAGEMAKER TRAINING" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Wait for new API Gateway deployment
Write-Host "Step 1: Waiting for API Gateway deployment..." -ForegroundColor Yellow
Write-Host ""
Write-Host "The new image with model-init routes is being deployed." -ForegroundColor White
Write-Host "This takes about 5-10 minutes." -ForegroundColor White
Write-Host ""

$deploymentReady = $false
$attempt = 0
$maxAttempts = 30

while (-not $deploymentReady -and $attempt -lt $maxAttempts) {
    $attempt++
    $timestamp = Get-Date -Format "HH:mm:ss"
    
    try {
        # Try to access the new endpoint
        $testResponse = Invoke-RestMethod -Uri "$ALB_URL/model-init/bootstrap" -Method POST -TimeoutSec 5 -ErrorAction Stop
        
        # If we get here (even with an error response), the endpoint exists!
        Write-Host "✅ [$timestamp] API Gateway deployment complete!" -ForegroundColor Green
        $deploymentReady = $true
        
    } catch {
        $errorMessage = $_.Exception.Message
        
        if ($errorMessage -like "*404*" -or $errorMessage -like "*Not Found*") {
            Write-Host "⏳ [$timestamp] Attempt $attempt/$maxAttempts - Waiting for deployment..." -ForegroundColor Cyan
            Start-Sleep -Seconds 20
        } elseif ($errorMessage -like "*bucket*" -or $errorMessage -like "*Training data*") {
            # This means the endpoint exists but training data isn't ready - that's OK!
            Write-Host "✅ [$timestamp] API Gateway deployment complete!" -ForegroundColor Green
            $deploymentReady = $true
        } else {
            Write-Host "⏳ [$timestamp] Attempt $attempt/$maxAttempts - $errorMessage" -ForegroundColor Yellow
            Start-Sleep -Seconds 20
        }
    }
}

if (-not $deploymentReady) {
    Write-Host ""
    Write-Host "❌ Deployment taking longer than expected." -ForegroundColor Red
    Write-Host "Check GitHub Actions: https://github.com/HAR5HA-7663/ml-sentiment-feedback-loop/actions" -ForegroundColor Yellow
    Write-Host "Then run: .\trigger-training.ps1" -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Step 2: Starting SageMaker Training..." -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

try {
    $response = Invoke-RestMethod -Uri "$ALB_URL/model-init/bootstrap" -Method POST -TimeoutSec 30
    
    if ($response.error) {
        Write-Host "❌ Error: $($response.error)" -ForegroundColor Red
        
        if ($response.error -like "*Training data not found*") {
            Write-Host ""
            Write-Host "The training data hasn't been uploaded to S3 yet." -ForegroundColor Yellow
            Write-Host "This should happen automatically via CI/CD." -ForegroundColor Yellow
            Write-Host ""
            Write-Host "Check if train_data.csv exists in S3:" -ForegroundColor White
            Write-Host "aws s3 ls s3://ml-sentiment-data-143519759870/" -ForegroundColor Cyan
        }
        
        exit 1
    }
    
    Write-Host "✅ Training Job Started!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Job Details:" -ForegroundColor Cyan
    Write-Host "  Job Name: $($response.training_job_name)" -ForegroundColor White
    Write-Host "  Status: $($response.status)" -ForegroundColor White
    Write-Host "  Estimated Time: $($response.estimated_time)" -ForegroundColor White
    Write-Host ""
    
    $JOB_NAME = $response.training_job_name
    $JOB_NAME | Out-File -FilePath "job-name.txt" -NoNewline
    
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "Step 3: Monitoring Training Progress..." -ForegroundColor Yellow
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Training will take approximately 15-20 minutes." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "View progress in AWS Console:" -ForegroundColor Yellow
    Write-Host "https://console.aws.amazon.com/sagemaker (Region: us-east-2)" -ForegroundColor Cyan
    Write-Host ""
    
    $completed = $false
    $checkCount = 0
    
    while (-not $completed) {
        $checkCount++
        Start-Sleep -Seconds 60
        
        try {
            $status = Invoke-RestMethod -Uri "$ALB_URL/model-init/status/$JOB_NAME" -TimeoutSec 10
            
            $timestamp = Get-Date -Format "HH:mm:ss"
            
            if ($status.status -eq "Completed") {
                Write-Host ""
                Write-Host "✅ [$timestamp] Training Complete!" -ForegroundColor Green
                Write-Host ""
                Write-Host "Model Artifacts: $($status.model_artifacts)" -ForegroundColor Cyan
                $completed = $true
                
                Write-Host ""
                Write-Host "================================================================" -ForegroundColor Cyan
                Write-Host "Step 4: Deploying Model to Endpoint..." -ForegroundColor Yellow
                Write-Host "================================================================" -ForegroundColor Cyan
                Write-Host ""
                
                $deployResponse = Invoke-RestMethod -Uri "$ALB_URL/model-init/deploy/$JOB_NAME" -Method POST -TimeoutSec 30
                
                Write-Host "✅ Deployment Started!" -ForegroundColor Green
                Write-Host "  Endpoint: $($deployResponse.endpoint_name)" -ForegroundColor White
                Write-Host "  Model: $($deployResponse.model_name)" -ForegroundColor White
                Write-Host "  Estimated Time: $($deployResponse.estimated_time)" -ForegroundColor White
                Write-Host ""
                
                Write-Host "================================================================" -ForegroundColor Cyan
                Write-Host "Step 5: Waiting for Endpoint..." -ForegroundColor Yellow
                Write-Host "================================================================" -ForegroundColor Cyan
                Write-Host ""
                
                $endpointReady = $false
                while (-not $endpointReady) {
                    Start-Sleep -Seconds 30
                    $endpointStatus = Invoke-RestMethod -Uri "$ALB_URL/model-init/endpoint-status" -TimeoutSec 10
                    
                    $timestamp = Get-Date -Format "HH:mm:ss"
                    Write-Host "⏳ [$timestamp] Endpoint Status: $($endpointStatus.status)" -ForegroundColor Cyan
                    
                    if ($endpointStatus.status -eq "InService") {
                        Write-Host ""
                        Write-Host "✅ Endpoint Ready!" -ForegroundColor Green
                        $endpointReady = $true
                    } elseif ($endpointStatus.status -eq "Failed") {
                        Write-Host ""
                        Write-Host "❌ Endpoint deployment failed!" -ForegroundColor Red
                        exit 1
                    }
                }
                
                Write-Host ""
                Write-Host "================================================================" -ForegroundColor Green
                Write-Host "  🎉 COMPLETE! ML SYSTEM IS READY! 🎉" -ForegroundColor Green
                Write-Host "================================================================" -ForegroundColor Green
                Write-Host ""
                Write-Host "Test your model now:" -ForegroundColor Yellow
                Write-Host ""
                Write-Host "# Make a prediction" -ForegroundColor Cyan
                Write-Host '$body = @{text = "This is amazing!"} | ConvertTo-Json' -ForegroundColor White
                Write-Host "Invoke-RestMethod -Uri `"$ALB_URL/predict`" -Method POST -Body `$body -ContentType `"application/json`"" -ForegroundColor White
                Write-Host ""
                Write-Host "# Submit feedback" -ForegroundColor Cyan
                Write-Host '$feedback = @{text = "Great product!"; model_prediction = "positive"; user_label = "positive"} | ConvertTo-Json' -ForegroundColor White
                Write-Host "Invoke-RestMethod -Uri `"$ALB_URL/feedback`" -Method POST -Body `$feedback -ContentType `"application/json`"" -ForegroundColor White
                Write-Host ""
                Write-Host "# Run evaluation" -ForegroundColor Cyan
                Write-Host "Invoke-RestMethod -Uri `"$ALB_URL/evaluate`" -Method POST" -ForegroundColor White
                Write-Host ""
                
            } elseif ($status.status -eq "Failed") {
                Write-Host ""
                Write-Host "❌ [$timestamp] Training Failed!" -ForegroundColor Red
                Write-Host "Reason: $($status.failure_reason)" -ForegroundColor Red
                $completed = $true
                exit 1
            } elseif ($status.status -eq "InProgress") {
                Write-Host "⏳ [$timestamp] Training in progress... (Check #$checkCount)" -ForegroundColor Cyan
            } else {
                Write-Host "⏳ [$timestamp] Status: $($status.status)" -ForegroundColor Yellow
            }
            
        } catch {
            Write-Host "⚠️ Error checking status: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
    
} catch {
    Write-Host ""
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    if ($_.ErrorDetails.Message) {
        $errorDetails = $_.ErrorDetails.Message | ConvertFrom-Json
        Write-Host "Details: $($errorDetails.error -or $errorDetails.detail)" -ForegroundColor Yellow
    }
    exit 1
}
