# Monitor SageMaker Training

$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"
$JOB_NAME = Get-Content "job-name.txt" -ErrorAction SilentlyContinue

if (-not $JOB_NAME) {
    Write-Host "❌ No job name found in job-name.txt" -ForegroundColor Red
    Write-Host "Run .\wait-and-train.ps1 first to start training" -ForegroundColor Yellow
    exit 1
}

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  MONITORING SAGEMAKER TRAINING" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Job: $JOB_NAME" -ForegroundColor White
Write-Host "Estimated time: 15-20 minutes" -ForegroundColor Yellow
Write-Host ""
Write-Host "View in AWS Console:" -ForegroundColor Cyan
Write-Host "https://console.aws.amazon.com/sagemaker (Region: us-east-2)" -ForegroundColor White
Write-Host ""
Write-Host "Checking status every minute..." -ForegroundColor Yellow
Write-Host ""

$completed = $false

while (-not $completed) {
    try {
        $status = Invoke-RestMethod -Uri "$ALB_URL/model-init/status/$JOB_NAME" -TimeoutSec 10
        
        $timestamp = Get-Date -Format "HH:mm:ss"
        
        if ($status.status -eq "Completed") {
            Write-Host ""
            Write-Host "✅ [$timestamp] TRAINING COMPLETE!" -ForegroundColor Green
            Write-Host ""
            Write-Host "Model artifacts: $($status.model_artifacts)" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "================================================================" -ForegroundColor Green
            Write-Host "  NEXT STEP: Deploy Model to Endpoint" -ForegroundColor Yellow
            Write-Host "================================================================" -ForegroundColor Green
            Write-Host ""
            Write-Host "Run this command:" -ForegroundColor White
            Write-Host ""
            Write-Host "  Invoke-RestMethod -Uri `"$ALB_URL/model-init/deploy/$JOB_NAME`" -Method POST | ConvertTo-Json" -ForegroundColor Cyan
            Write-Host ""
            $completed = $true
            
        } elseif ($status.status -eq "Failed") {
            Write-Host ""
            Write-Host "❌ [$timestamp] TRAINING FAILED!" -ForegroundColor Red
            Write-Host ""
            Write-Host "Reason: $($status.failure_reason)" -ForegroundColor Yellow
            Write-Host ""
            $completed = $true
            exit 1
            
        } elseif ($status.status -eq "InProgress" -or $status.status -eq "Starting") {
            Write-Host "⏳ [$timestamp] Training in progress..." -ForegroundColor Cyan
            
        } else {
            Write-Host "⏳ [$timestamp] Status: $($status.status)" -ForegroundColor Yellow
        }
        
    } catch {
        $timestamp = Get-Date -Format "HH:mm:ss"
        Write-Host "⚠️ [$timestamp] Error checking status (service may be restarting)" -ForegroundColor Yellow
    }
    
    if (-not $completed) {
        Start-Sleep -Seconds 60
    }
}
