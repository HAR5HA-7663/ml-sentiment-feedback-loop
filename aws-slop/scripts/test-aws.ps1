# ML Sentiment Deployment Test Script
$BASE_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "ML Sentiment Deployment Tests" -ForegroundColor Cyan
Write-Host "=================================`n" -ForegroundColor Cyan

# Test 1: Health Check
Write-Host "[1] Testing API Gateway Health..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$BASE_URL/health" -Method Get -ErrorAction Stop
    Write-Host "SUCCESS - API Gateway is running" -ForegroundColor Green
    Write-Host "Gateway Status: $($health.gateway)`n" -ForegroundColor Gray
} catch {
    Write-Host "FAILED - $_`n" -ForegroundColor Red
}

# Test 2: Predict Sentiment
Write-Host "[2] Testing Sentiment Prediction..." -ForegroundColor Yellow
try {
    $body = @{ text = "This product is amazing!" } | ConvertTo-Json
    $pred = Invoke-RestMethod -Uri "$BASE_URL/predict-sentiment" -Method Post -Body $body -ContentType "application/json" -ErrorAction Stop
    Write-Host "SUCCESS - Prediction endpoint working" -ForegroundColor Green
    Write-Host "Response: $($pred | ConvertTo-Json -Compress)`n" -ForegroundColor Gray
} catch {
    Write-Host "FAILED - $_`n" -ForegroundColor Red
}

# Test 3: Model Registry
Write-Host "[3] Testing Model Registry..." -ForegroundColor Yellow
try {
    $models = Invoke-RestMethod -Uri "$BASE_URL/models" -Method Get -ErrorAction Stop
    Write-Host "SUCCESS - Model registry accessible" -ForegroundColor Green
    Write-Host "Response: $($models | ConvertTo-Json -Compress)`n" -ForegroundColor Gray
} catch {
    Write-Host "FAILED - $_`n" -ForegroundColor Red
}

# Test 4: Submit Feedback
Write-Host "[4] Testing Feedback Submission..." -ForegroundColor Yellow
try {
    $feedback = @{
        text = "Great product!"
        prediction = "positive"
        actual_sentiment = "positive"
    } | ConvertTo-Json
    $result = Invoke-RestMethod -Uri "$BASE_URL/submit-feedback" -Method Post -Body $feedback -ContentType "application/json" -ErrorAction Stop
    Write-Host "SUCCESS - Feedback endpoint working" -ForegroundColor Green
    Write-Host "Response: $($result | ConvertTo-Json -Compress)`n" -ForegroundColor Gray
} catch {
    Write-Host "FAILED - $_`n" -ForegroundColor Red
}

# Test 5: ECS Services Status
Write-Host "[5] Checking ECS Services..." -ForegroundColor Yellow
Write-Host "Running AWS CLI command...`n" -ForegroundColor Gray
aws ecs describe-services --cluster ml-sentiment-cluster --services ml-sentiment-api-gateway-service ml-sentiment-inference-service ml-sentiment-feedback-service ml-sentiment-model-registry-service --profile default --region us-east-2 --query "services[*].[serviceName,status,runningCount,desiredCount]" --output table

Write-Host "`n=================================" -ForegroundColor Cyan
Write-Host "Testing Complete!" -ForegroundColor Cyan
Write-Host "=================================`n" -ForegroundColor Cyan
Write-Host "Your deployment URL: $BASE_URL" -ForegroundColor White
Write-Host "`nNext Steps:" -ForegroundColor White
Write-Host "1. All services are running on AWS ECS" -ForegroundColor Gray
Write-Host "2. Use the endpoints above for your demo" -ForegroundColor Gray
Write-Host "3. Monitor costs in AWS Console" -ForegroundColor Gray
Write-Host "4. Auto-shutdown active (11 PM - 7 AM EST)`n" -ForegroundColor Gray
