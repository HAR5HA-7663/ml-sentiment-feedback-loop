# Quick check if endpoint is ready and test

$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  CHECKING ENDPOINT STATUS" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$endpoint = aws sagemaker describe-endpoint --endpoint-name ml-sentiment-endpoint --region us-east-2 | ConvertFrom-Json
$status = $endpoint.EndpointStatus
$config = $endpoint.EndpointConfigName

Write-Host "Status: $status" -ForegroundColor $(if ($status -eq "InService") { "Green" } elseif ($status -eq "Updating") { "Yellow" } else { "Red" })
Write-Host "Config: $config" -ForegroundColor Cyan
Write-Host ""

if ($status -eq "InService" -and $config -eq "ml-sentiment-endpoint-config-20251213-224111") {
    Write-Host "✅ CORRECT MODEL IS ACTIVE!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Testing predictions..." -ForegroundColor Cyan
    Write-Host ""
    
    try {
        $result1 = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body '{"text":"This product is amazing! I love it so much!"}' -ContentType "application/json" -TimeoutSec 10
        Write-Host "Test 1: 'This product is amazing!'" -ForegroundColor Yellow
        Write-Host "  Sentiment: $($result1.sentiment -or $result1.label)" -ForegroundColor Green
        Write-Host "  Confidence: $([math]::Round($result1.confidence * 100, 2))%" -ForegroundColor Cyan
        Write-Host ""
        
        $result2 = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body '{"text":"Terrible product! Complete waste of money!"}' -ContentType "application/json" -TimeoutSec 10
        Write-Host "Test 2: 'Terrible product! Waste of money!'" -ForegroundColor Yellow
        Write-Host "  Sentiment: $($result2.sentiment -or $result2.label)" -ForegroundColor Green
        Write-Host "  Confidence: $([math]::Round($result2.confidence * 100, 2))%" -ForegroundColor Cyan
        Write-Host ""
        
        $sentiment1 = $result1.sentiment -or $result1.label
        $sentiment2 = $result2.sentiment -or $result2.label
        
        if ($sentiment1 -eq "positive" -and $sentiment2 -eq "negative") {
            Write-Host "================================================================" -ForegroundColor Green
            Write-Host "  🎉 ML SYSTEM IS WORKING PERFECTLY!" -ForegroundColor Green
            Write-Host "================================================================" -ForegroundColor Green
            Write-Host ""
            Write-Host "✅ SageMaker Training: Complete" -ForegroundColor Green
            Write-Host "✅ Model Deployment: Active" -ForegroundColor Green
            Write-Host "✅ Real-time Predictions: Working" -ForegroundColor Green
            Write-Host ""
            Write-Host "Open test-api.html in your browser to try it!" -ForegroundColor Yellow
        } else {
            Write-Host "⚠️ Predictions received but may need adjustment" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ Error testing: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Endpoint may still be initializing..." -ForegroundColor Yellow
    }
    
} elseif ($status -eq "Updating") {
    Write-Host "⏳ Endpoint is updating..." -ForegroundColor Yellow
    Write-Host "This takes 3-5 minutes. Run this script again in a few minutes!" -ForegroundColor White
    Write-Host ""
    Write-Host "You can monitor in AWS Console:" -ForegroundColor Cyan
    Write-Host "  https://console.aws.amazon.com/sagemaker" -ForegroundColor White
    Write-Host "  Region: us-east-2" -ForegroundColor White
    Write-Host "  Endpoints → ml-sentiment-endpoint" -ForegroundColor White
} else {
    Write-Host "⚠️ Endpoint status: $status" -ForegroundColor Yellow
    Write-Host "Config: $config" -ForegroundColor Yellow
}
