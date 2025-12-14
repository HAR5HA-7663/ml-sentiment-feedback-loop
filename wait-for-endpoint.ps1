# Wait for SageMaker Endpoint to be ready

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  WAITING FOR SAGEMAKER ENDPOINT" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Endpoint: ml-sentiment-endpoint" -ForegroundColor Yellow
Write-Host "Expected time: 3-8 minutes" -ForegroundColor Yellow
Write-Host ""
Write-Host "Checking status every 30 seconds..." -ForegroundColor White
Write-Host ""

$maxAttempts = 20
$attempt = 0

while ($attempt -lt $maxAttempts) {
    $attempt++
    $timestamp = Get-Date -Format "HH:mm:ss"
    
    try {
        $endpoint = aws sagemaker describe-endpoint --endpoint-name ml-sentiment-endpoint --region us-east-2 | ConvertFrom-Json
        $status = $endpoint.EndpointStatus
        
        if ($status -eq "InService") {
            Write-Host ""
            Write-Host "================================================================" -ForegroundColor Green
            Write-Host "  ✅ ENDPOINT IS LIVE!" -ForegroundColor Green
            Write-Host "================================================================" -ForegroundColor Green
            Write-Host ""
            Write-Host "Now testing predictions..." -ForegroundColor Cyan
            Write-Host ""
            
            # Test predictions
            $ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"
            
            Write-Host "Test 1: Positive Review" -ForegroundColor Yellow
            $result1 = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body '{"text":"This product is amazing! Best purchase ever!"}' -ContentType "application/json"
            Write-Host "  Text: 'This product is amazing! Best purchase ever!'" -ForegroundColor White
            Write-Host "  Sentiment: $($result1.sentiment -or $result1.label)" -ForegroundColor Green
            Write-Host "  Confidence: $($result1.confidence)" -ForegroundColor Cyan
            Write-Host ""
            
            Write-Host "Test 2: Negative Review" -ForegroundColor Yellow
            $result2 = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body '{"text":"Terrible quality, complete waste of money"}' -ContentType "application/json"
            Write-Host "  Text: 'Terrible quality, complete waste of money'" -ForegroundColor White
            Write-Host "  Sentiment: $($result2.sentiment -or $result2.label)" -ForegroundColor Green
            Write-Host "  Confidence: $($result2.confidence)" -ForegroundColor Cyan
            Write-Host ""
            
            Write-Host "================================================================" -ForegroundColor Green
            Write-Host "  🎉 YOUR ML SYSTEM IS FULLY OPERATIONAL!" -ForegroundColor Green
            Write-Host "================================================================" -ForegroundColor Green
            Write-Host ""
            Write-Host "Open test-api.html in your browser to interact with it!" -ForegroundColor Yellow
            Write-Host ""
            break
        }
        elseif ($status -eq "Failed") {
            Write-Host ""
            Write-Host "❌ [$timestamp] ENDPOINT UPDATE FAILED!" -ForegroundColor Red
            Write-Host ""
            Write-Host "Failure reason: $($endpoint.FailureReason)" -ForegroundColor Yellow
            break
        }
        else {
            Write-Host "[$timestamp] Status: $status (attempt $attempt/$maxAttempts)" -ForegroundColor Cyan
        }
    }
    catch {
        Write-Host "[$timestamp] Error checking status: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    
    Start-Sleep -Seconds 30
}

if ($attempt -ge $maxAttempts) {
    Write-Host ""
    Write-Host "⚠️ Timeout reached. Check AWS Console:" -ForegroundColor Yellow
    Write-Host "https://console.aws.amazon.com/sagemaker" -ForegroundColor Cyan
    Write-Host "Region: us-east-2" -ForegroundColor White
}
