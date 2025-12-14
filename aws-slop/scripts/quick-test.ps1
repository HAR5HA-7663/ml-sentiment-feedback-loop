# Quick test - wait for endpoint and test

$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  TESTING ML PREDICTIONS" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Check endpoint status
$endpoint = aws sagemaker describe-endpoint --endpoint-name ml-sentiment-endpoint --region us-east-2 | ConvertFrom-Json
$status = $endpoint.EndpointStatus
$config = $endpoint.EndpointConfigName

Write-Host "Endpoint Status: $status" -ForegroundColor $(if ($status -eq "InService") { "Green" } else { "Yellow" })
Write-Host "Config: $config" -ForegroundColor Cyan
Write-Host ""

if ($status -eq "InService" -and $config -eq "ml-sentiment-endpoint-config-20251213-224111") {
    Write-Host "✅ CORRECT MODEL IS ACTIVE!" -ForegroundColor Green
    Write-Host ""
} elseif ($status -eq "Updating") {
    Write-Host "⏳ Endpoint is updating... This takes 3-5 minutes" -ForegroundColor Yellow
    Write-Host "Run this script again in a few minutes!" -ForegroundColor White
    exit 0
} else {
    Write-Host "⚠️ Endpoint not ready yet" -ForegroundColor Yellow
    Write-Host ""
}

# Test predictions
Write-Host "Testing predictions..." -ForegroundColor Cyan
Write-Host ""

try {
    Write-Host "Test 1: Positive Review" -ForegroundColor Yellow
    $result1 = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body '{"text":"This product is amazing! I love it so much! Best purchase ever!"}' -ContentType "application/json"
    Write-Host "  Input: 'This product is amazing! I love it so much!'" -ForegroundColor White
    Write-Host "  Sentiment: $($result1.sentiment -or $result1.label)" -ForegroundColor Green
    Write-Host "  Confidence: $([math]::Round($result1.confidence * 100, 2))%" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "Test 2: Negative Review" -ForegroundColor Yellow
    $result2 = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body '{"text":"Terrible product! Complete waste of money! Do not buy!"}' -ContentType "application/json"
    Write-Host "  Input: 'Terrible product! Complete waste of money!'" -ForegroundColor White
    Write-Host "  Sentiment: $($result2.sentiment -or $result2.label)" -ForegroundColor Green
    Write-Host "  Confidence: $([math]::Round($result2.confidence * 100, 2))%" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "Test 3: Neutral Review" -ForegroundColor Yellow
    $result3 = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body '{"text":"The product arrived on time as expected"}' -ContentType "application/json"
    Write-Host "  Input: 'The product arrived on time'" -ForegroundColor White
    Write-Host "  Sentiment: $($result3.sentiment -or $result3.label)" -ForegroundColor Green
    Write-Host "  Confidence: $([math]::Round($result3.confidence * 100, 2))%" -ForegroundColor Cyan
    Write-Host ""
    
    # Check if predictions are working correctly
    $sentiment1 = $result1.sentiment -or $result1.label
    $sentiment2 = $result2.sentiment -or $result2.label
    
    if ($sentiment1 -eq "positive" -and $sentiment2 -eq "negative") {
        Write-Host "================================================================" -ForegroundColor Green
        Write-Host "  🎉 ML SYSTEM IS WORKING PERFECTLY!" -ForegroundColor Green
        Write-Host "================================================================" -ForegroundColor Green
    } elseif ($sentiment1 -eq "True" -or $sentiment1 -eq "neutral") {
        Write-Host "================================================================" -ForegroundColor Yellow
        Write-Host "  ⚠️ Predictions are returning but may need endpoint update" -ForegroundColor Yellow
        Write-Host "================================================================" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "The endpoint is responding but may still be using old model." -ForegroundColor White
        Write-Host "Wait a few more minutes and test again!" -ForegroundColor White
    } else {
        Write-Host "================================================================" -ForegroundColor Yellow
        Write-Host "  Predictions received - check results above" -ForegroundColor Yellow
        Write-Host "================================================================" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "❌ Error testing predictions: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "The endpoint may still be updating. Try again in a few minutes." -ForegroundColor Yellow
}
