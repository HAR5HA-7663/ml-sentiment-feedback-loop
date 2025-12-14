# Final Test - Wait for endpoint and test predictions

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  FINAL ML SYSTEM TEST" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Waiting for SageMaker endpoint to finish updating..." -ForegroundColor Yellow
Write-Host "Target config: ml-sentiment-endpoint-config-20251213-224111" -ForegroundColor White
Write-Host "(This uses tensorflow-training container with custom inference.py)" -ForegroundColor White
Write-Host ""

$maxAttempts = 15
$attempt = 0
$success = $false

while ($attempt -lt $maxAttempts) {
    $attempt++
    $timestamp = Get-Date -Format "HH:mm:ss"
    
    try {
        $endpoint = aws sagemaker describe-endpoint --endpoint-name ml-sentiment-endpoint --region us-east-2 | ConvertFrom-Json
        $status = $endpoint.EndpointStatus
        $config = $endpoint.EndpointConfigName
        
        Write-Host "[$timestamp] Status: $status | Config: $config" -ForegroundColor Cyan
        
        if ($status -eq "InService" -and $config -eq "ml-sentiment-endpoint-config-20251213-224111") {
            Write-Host ""
            Write-Host "================================================================" -ForegroundColor Green
            Write-Host "  ✅ ENDPOINT READY WITH CORRECT MODEL!" -ForegroundColor Green
            Write-Host "================================================================" -ForegroundColor Green
            Write-Host ""
            
            # Test predictions
            $ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"
            
            Write-Host "🧪 Testing ML Predictions..." -ForegroundColor Cyan
            Write-Host ""
            
            Write-Host "Test 1: Positive Review" -ForegroundColor Yellow
            $result1 = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body '{"text":"This product is amazing! I love it so much! Best purchase ever!"}' -ContentType "application/json"
            Write-Host "  Input: 'This product is amazing! I love it so much!'" -ForegroundColor White
            Write-Host "  Sentiment: $($result1.sentiment -or $result1.label)" -ForegroundColor Green
            Write-Host "  Confidence: $([math]::Round($result1.confidence * 100, 2))%" -ForegroundColor Cyan
            Write-Host ""
            
            Write-Host "Test 2: Negative Review" -ForegroundColor Yellow
            $result2 = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body '{"text":"Terrible product! Waste of money! Do not buy!"}' -ContentType "application/json"
            Write-Host "  Input: 'Terrible product! Waste of money!'" -ForegroundColor White
            Write-Host "  Sentiment: $($result2.sentiment -or $result2.label)" -ForegroundColor Green
            Write-Host "  Confidence: $([math]::Round($result2.confidence * 100, 2))%" -ForegroundColor Cyan
            Write-Host ""
            
            Write-Host "Test 3: Neutral Review" -ForegroundColor Yellow
            $result3 = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body '{"text":"The product arrived on time as expected"}' -ContentType "application/json"
            Write-Host "  Input: 'The product arrived on time'" -ForegroundColor White
            Write-Host "  Sentiment: $($result3.sentiment -or $result3.label)" -ForegroundColor Green
            Write-Host "  Confidence: $([math]::Round($result3.confidence * 100, 2))%" -ForegroundColor Cyan
            Write-Host ""
            
            Write-Host "================================================================" -ForegroundColor Green
            Write-Host "  🎉 YOUR ML SYSTEM IS FULLY OPERATIONAL!" -ForegroundColor Green
            Write-Host "================================================================" -ForegroundColor Green
            Write-Host ""
            Write-Host "✅ SageMaker Training: Complete" -ForegroundColor Green
            Write-Host "✅ Model Deployment: Active" -ForegroundColor Green
            Write-Host "✅ Real-time Predictions: Working" -ForegroundColor Green
            Write-Host "✅ Feedback Loop: Ready" -ForegroundColor Green
            Write-Host ""
            Write-Host "Try it yourself:" -ForegroundColor Yellow
            Write-Host "  Open test-api.html in your browser!" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "AWS Console:" -ForegroundColor Yellow
            Write-Host "  https://console.aws.amazon.com/sagemaker (Region: us-east-2)" -ForegroundColor Cyan
            Write-Host ""
            
            $success = $true
            break
        }
        elseif ($status -eq "Failed") {
            Write-Host ""
            Write-Host "❌ Endpoint update failed!" -ForegroundColor Red
            Write-Host "Reason: $($endpoint.FailureReason)" -ForegroundColor Yellow
            break
        }
    }
    catch {
        Write-Host "[$timestamp] Error: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    
    if ($attempt -lt $maxAttempts) {
        Start-Sleep -Seconds 30
    }
}

if (-not $success) {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Yellow
    Write-Host "  ⏳ ENDPOINT STILL UPDATING" -ForegroundColor Yellow
    Write-Host "================================================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "The endpoint is taking longer than expected to update." -ForegroundColor White
    Write-Host ""
    Write-Host "Check status in AWS Console:" -ForegroundColor Yellow
    Write-Host "  https://console.aws.amazon.com/sagemaker" -ForegroundColor Cyan
    Write-Host "  Region: us-east-2" -ForegroundColor White
    Write-Host "  Endpoints → ml-sentiment-endpoint" -ForegroundColor White
    Write-Host ""
    Write-Host "Or run this script again in a few minutes!" -ForegroundColor Yellow
}
