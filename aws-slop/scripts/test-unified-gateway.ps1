# Test Unified API Gateway
$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "   TESTING UNIFIED API GATEWAY" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Health Check
Write-Host "1. Health Check:" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$ALB_URL/health" -Method GET
    $response | ConvertTo-Json -Depth 5
    Write-Host ""
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

# 2. Predict
Write-Host "2. Predict (through API Gateway):" -ForegroundColor Yellow
try {
    $body = @{ text = "I absolutely love this amazing product!" } | ConvertTo-Json
    $response = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body $body -ContentType "application/json"
    $response | ConvertTo-Json
    Write-Host ""
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Response: $($_.ErrorDetails.Message)" -ForegroundColor Red
    Write-Host ""
}

# 3. Models
Write-Host "3. Models:" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$ALB_URL/models" -Method GET
    $response | ConvertTo-Json
    Write-Host ""
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

# 4. Feedback
Write-Host "4. Feedback:" -ForegroundColor Yellow
try {
    $body = @{
        text = "Great product!"
        model_prediction = "positive"
        user_label = "positive"
    } | ConvertTo-Json
    $response = Invoke-RestMethod -Uri "$ALB_URL/feedback" -Method POST -Body $body -ContentType "application/json"
    $response | ConvertTo-Json
    Write-Host ""
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "If /predict works, the unified API Gateway is ready!" -ForegroundColor Green
Write-Host "All traffic now routes through one entry point." -ForegroundColor Green
