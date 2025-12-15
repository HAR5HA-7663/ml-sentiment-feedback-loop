# Monitor SageMaker Training Job
# Usage: .\monitor-training.ps1 -JobName "ml-sentiment-training-20251214-223057"

param(
    [Parameter(Mandatory = $true)]
    [string]$JobName,
    
    [string]$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"
)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Monitoring SageMaker Training Job" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Job Name: $JobName" -ForegroundColor White
Write-Host ""

$completed = $false
$checkCount = 0

while (-not $completed) {
    $checkCount++
    Start-Sleep -Seconds 30
    
    try {
        $status = Invoke-RestMethod -Uri "$ALB_URL/model-init/status/$JobName" -TimeoutSec 10
        
        $timestamp = Get-Date -Format "HH:mm:ss"
        
        if ($status.status -eq "Completed") {
            Write-Host "✅ [$timestamp] Training Complete!" -ForegroundColor Green
            Write-Host ""
            Write-Host "Model Artifacts: $($status.model_artifacts)" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "Next step: Deploy the model" -ForegroundColor Yellow
            Write-Host "Command: Invoke-RestMethod -Uri `"$ALB_URL/model-init/deploy/$JobName`" -Method POST" -ForegroundColor White
            $completed = $true
            
        }
        elseif ($status.status -eq "Failed") {
            Write-Host "❌ [$timestamp] Training Failed!" -ForegroundColor Red
            Write-Host "Reason: $($status.failure_reason)" -ForegroundColor Red
            $completed = $true
            exit 1
            
        }
        elseif ($status.status -eq "InProgress") {
            Write-Host "⏳ [$timestamp] Training in progress... (Check #$checkCount)" -ForegroundColor Cyan
            if ($status.estimated_time_remaining) {
                Write-Host "   Estimated time remaining: $($status.estimated_time_remaining)" -ForegroundColor Gray
            }
        }
        else {
            Write-Host "⏳ [$timestamp] Status: $($status.status)" -ForegroundColor Yellow
        }
        
    }
    catch {
        Write-Host "⚠️ Error checking status: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}
