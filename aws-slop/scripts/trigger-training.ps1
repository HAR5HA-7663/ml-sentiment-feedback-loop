# Trigger SageMaker Training Script
# This script starts the ML training process and monitors it

$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  SageMaker Training Trigger Script" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Wait for API Gateway to be ready
Write-Host "Step 1: Checking if API Gateway is ready..." -ForegroundColor Yellow
Write-Host ""

$maxAttempts = 20
$attempt = 0
$ready = $false

while ($attempt -lt $maxAttempts -and -not $ready) {
    $attempt++
    try {
        $health = Invoke-RestMethod -Uri "$ALB_URL/health" -TimeoutSec 5
        if ($health.overall -eq "healthy") {
            Write-Host "✅ API Gateway is ready!" -ForegroundColor Green
            $ready = $true
        } else {
            Write-Host "⏳ Attempt $attempt/$maxAttempts - Services starting..." -ForegroundColor Yellow
            Start-Sleep -Seconds 15
        }
    } catch {
        Write-Host "⏳ Attempt $attempt/$maxAttempts - Waiting for deployment..." -ForegroundColor Yellow
        Start-Sleep -Seconds 15
    }
}

if (-not $ready) {
    Write-Host "❌ API Gateway not ready. Please wait a few more minutes and try again." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Step 2: Starting SageMaker Training..." -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

try {
    $response = Invoke-RestMethod -Uri "$ALB_URL/model-init/bootstrap" -Method POST -TimeoutSec 30
    
    Write-Host "✅ Training Job Started!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Job Details:" -ForegroundColor Cyan
    Write-Host "  Job Name: $($response.training_job_name)" -ForegroundColor White
    Write-Host "  Status: $($response.status)" -ForegroundColor White
    Write-Host "  Estimated Time: $($response.estimated_time)" -ForegroundColor White
    Write-Host ""
    
    $JOB_NAME = $response.training_job_name
    $JOB_NAME | Out-File -FilePath "job-name.txt" -NoNewline
    
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "Step 3: Monitoring Training Progress..." -ForegroundColor Yellow
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "This will take approximately 15-20 minutes." -ForegroundColor Yellow
    Write-Host "You can also view progress in AWS Console:" -ForegroundColor Yellow
    Write-Host "https://console.aws.amazon.com/sagemaker" -ForegroundColor Cyan
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
                Write-Host "✅ [$timestamp] Training Complete!" -ForegroundColor Green
                Write-Host ""
                Write-Host "Model Artifacts: $($status.model_artifacts)" -ForegroundColor Cyan
                $completed = $true
                
                Write-Host ""
                Write-Host "================================================" -ForegroundColor Cyan
                Write-Host "Step 4: Deploying Model to Endpoint..." -ForegroundColor Yellow
                Write-Host "================================================" -ForegroundColor Cyan
                Write-Host ""
                
                $deployResponse = Invoke-RestMethod -Uri "$ALB_URL/model-init/deploy/$JOB_NAME" -Method POST -TimeoutSec 30
                
                Write-Host "✅ Deployment Started!" -ForegroundColor Green
                Write-Host "  Endpoint: $($deployResponse.endpoint_name)" -ForegroundColor White
                Write-Host "  Model: $($deployResponse.model_name)" -ForegroundColor White
                Write-Host "  Estimated Time: $($deployResponse.estimated_time)" -ForegroundColor White
                Write-Host ""
                
                Write-Host "================================================" -ForegroundColor Cyan
                Write-Host "Step 5: Waiting for Endpoint..." -ForegroundColor Yellow
                Write-Host "================================================" -ForegroundColor Cyan
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
                Write-Host "================================================" -ForegroundColor Green
                Write-Host "  🎉 COMPLETE! ML SYSTEM IS READY! 🎉" -ForegroundColor Green
                Write-Host "================================================" -ForegroundColor Green
                Write-Host ""
                Write-Host "Test your model:" -ForegroundColor Yellow
                Write-Host ""
                Write-Host '$body = @{text = "This is amazing!"} | ConvertTo-Json' -ForegroundColor White
                Write-Host "Invoke-RestMethod -Uri `"$ALB_URL/predict`" -Method POST -Body `$body -ContentType `"application/json`"" -ForegroundColor White
                Write-Host ""
                
            } elseif ($status.status -eq "Failed") {
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
    Write-Host "❌ Error starting training: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Response:" -ForegroundColor Yellow
    Write-Host $_.ErrorDetails.Message -ForegroundColor Red
    exit 1
}
