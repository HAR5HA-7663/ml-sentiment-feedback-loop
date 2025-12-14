# ML Sentiment Feedback Loop - AWS Deployment Testing Script
# Base URL
$BASE_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ML Sentiment Deployment Test Suite" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Test 1: API Gateway Health
Write-Host "[TEST 1] API Gateway Health Check" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/health" -Method Get
    Write-Host "✓ API Gateway is responding" -ForegroundColor Green
    Write-Host "  Status: $($response.gateway)" -ForegroundColor Gray
    Write-Host ""
} catch {
    Write-Host "✗ API Gateway health check failed: $_" -ForegroundColor Red
    Write-Host ""
}

# Test 2: Inference Service - Predict Sentiment
Write-Host "[TEST 2] Inference Service - Predict Sentiment" -ForegroundColor Yellow
try {
    $body = @{
        text = "This product is absolutely amazing! I love it!"
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri "$BASE_URL/predict-sentiment" -Method Post -Body $body -ContentType "application/json"
    Write-Host "✓ Prediction successful" -ForegroundColor Green
    Write-Host "  Response: $($response | ConvertTo-Json -Compress)" -ForegroundColor Gray
    Write-Host ""
} catch {
    $statusCode = $_.Exception.Response.StatusCode.value__
    Write-Host "✗ Prediction failed (HTTP $statusCode)" -ForegroundColor Red
    if ($_.ErrorDetails.Message) {
        Write-Host "  Error: $($_.ErrorDetails.Message)" -ForegroundColor Gray
    }
    Write-Host ""
}

# Test 3: Feedback Service
Write-Host "[TEST 3] Feedback Service - Submit Feedback" -ForegroundColor Yellow
try {
    $body = @{
        text = "Great product!"
        prediction = "positive"
        actual_sentiment = "positive"
        feedback_type = "correct"
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri "$BASE_URL/submit-feedback" -Method Post -Body $body -ContentType "application/json"
    Write-Host "✓ Feedback submitted successfully" -ForegroundColor Green
    Write-Host "  Response: $($response | ConvertTo-Json -Compress)" -ForegroundColor Gray
    Write-Host ""
} catch {
    $statusCode = $_.Exception.Response.StatusCode.value__
    Write-Host "✗ Feedback submission failed (HTTP $statusCode)" -ForegroundColor Red
    if ($_.ErrorDetails.Message) {
        Write-Host "  Error: $($_.ErrorDetails.Message)" -ForegroundColor Gray
    }
    Write-Host ""
}

# Test 4: Model Registry Service
Write-Host "[TEST 4] Model Registry - List Models" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/models" -Method Get
    Write-Host "✓ Model registry is responding" -ForegroundColor Green
    Write-Host "  Response: $($response | ConvertTo-Json -Compress)" -ForegroundColor Gray
    Write-Host ""
} catch {
    $statusCode = $_.Exception.Response.StatusCode.value__
    Write-Host "✗ Model registry failed (HTTP $statusCode)" -ForegroundColor Red
    if ($_.ErrorDetails.Message) {
        Write-Host "  Error: $($_.ErrorDetails.Message)" -ForegroundColor Gray
    }
    Write-Host ""
}

# Test 5: Check ECS Services
Write-Host "[TEST 5] ECS Services Status" -ForegroundColor Yellow
try {
    $services = aws ecs describe-services --cluster ml-sentiment-cluster `
        --services ml-sentiment-api-gateway-service ml-sentiment-inference-service ml-sentiment-feedback-service ml-sentiment-model-registry-service `
        --profile default --region us-east-2 --output json | ConvertFrom-Json
    
    foreach ($svc in $services.services) {
        $name = $svc.serviceName -replace "ml-sentiment-", ""
        $status = if ($svc.runningCount -eq $svc.desiredCount) { "✓" } else { "✗" }
        $color = if ($svc.runningCount -eq $svc.desiredCount) { "Green" } else { "Red" }
        Write-Host "  $status $name`: $($svc.runningCount)/$($svc.desiredCount) running" -ForegroundColor $color
    }
    Write-Host ""
} catch {
    Write-Host "✗ Failed to check ECS services: $_" -ForegroundColor Red
    Write-Host ""
}

# Test 6: Check S3 Buckets
Write-Host "[TEST 6] S3 Buckets" -ForegroundColor Yellow
try {
    $buckets = aws s3 ls --profile default --region us-east-2 2>&1 | Select-String "ml-sentiment"
    foreach ($bucket in $buckets) {
        Write-Host "  ✓ $bucket" -ForegroundColor Green
    }
    Write-Host ""
} catch {
    Write-Host "✗ Failed to list S3 buckets: $_" -ForegroundColor Red
    Write-Host ""
}

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Base URL: $BASE_URL" -ForegroundColor White
Write-Host ""
Write-Host "Available Endpoints:" -ForegroundColor White
Write-Host "  GET  /health            - API Gateway health" -ForegroundColor Gray
Write-Host "  POST /predict-sentiment - Sentiment prediction" -ForegroundColor Gray
Write-Host "  POST /submit-feedback   - Submit user feedback" -ForegroundColor Gray
Write-Host "  GET  /models            - List registered models" -ForegroundColor Gray
Write-Host "  POST /run-evaluation    - Run model evaluation" -ForegroundColor Gray
Write-Host "  POST /retrain           - Trigger model retraining" -ForegroundColor Gray
Write-Host ""
Write-Host "Cost Monitoring:" -ForegroundColor White
Write-Host "  Budget: `$180/month" -ForegroundColor Gray
Write-Host "  Auto-shutdown: 11 PM - 7 AM EST daily" -ForegroundColor Gray
Write-Host ""
