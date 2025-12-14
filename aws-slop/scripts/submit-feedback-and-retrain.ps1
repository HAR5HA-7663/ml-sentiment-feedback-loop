# Submit feedback and trigger retraining workflow

$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"

Write-Host "================================================================" -ForegroundColor Green
Write-Host "  FEEDBACK & RETRAINING WORKFLOW" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Step 1: Get a prediction
Write-Host "Step 1: Get Prediction" -ForegroundColor Cyan
$testText = "mehh not that great"
Write-Host "  Text: '$testText'" -ForegroundColor White

$prediction = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body (@{text=$testText} | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 20
$predictedLabel = $prediction.label -or $prediction.sentiment
$confidence = $prediction.confidence

Write-Host "  Prediction: $predictedLabel" -ForegroundColor Green
Write-Host "  Confidence: $([math]::Round($confidence * 100, 2))%" -ForegroundColor Cyan
Write-Host ""

# Step 2: Submit Feedback (correct the prediction)
Write-Host "Step 2: Submit Feedback" -ForegroundColor Cyan
Write-Host "  Model predicted: $predictedLabel" -ForegroundColor Yellow
Write-Host "  User corrects to: negative" -ForegroundColor Yellow

$feedback = @{
    text = $testText
    model_prediction = $predictedLabel
    user_label = "negative"  # User corrects the prediction
} | ConvertTo-Json

$feedbackResult = Invoke-RestMethod -Uri "$ALB_URL/feedback" -Method POST -Body $feedback -ContentType "application/json" -TimeoutSec 10
Write-Host "  ✅ Feedback submitted: $($feedbackResult.message)" -ForegroundColor Green
Write-Host "  Feedback ID: $($feedbackResult.id)" -ForegroundColor Cyan
Write-Host ""

# Step 3: Check if we have enough feedback for retraining
Write-Host "Step 3: Check Feedback Count" -ForegroundColor Cyan
Write-Host "  Note: Retraining requires at least 10 feedback samples" -ForegroundColor Yellow
Write-Host "  Current feedback count: (checking...)" -ForegroundColor Yellow
Write-Host ""

# Step 4: Trigger Retraining (if enough feedback)
Write-Host "Step 4: Trigger Retraining" -ForegroundColor Cyan
Write-Host "  ⚠️ Note: This will fail if < 10 feedback samples exist" -ForegroundColor Yellow
Write-Host "  Triggering retraining..." -ForegroundColor Yellow

try {
    $retrainResult = Invoke-RestMethod -Uri "$ALB_URL/retrain" -Method POST -ContentType "application/json" -TimeoutSec 300
    Write-Host "  ✅ Retraining completed!" -ForegroundColor Green
    Write-Host "  Model Version: $($retrainResult.version)" -ForegroundColor Cyan
    Write-Host "  Accuracy: $([math]::Round($retrainResult.accuracy * 100, 2))%" -ForegroundColor Cyan
    Write-Host "  Training Samples: $($retrainResult.training_samples)" -ForegroundColor Cyan
} catch {
    $errorMsg = $_.Exception.Message
    if ($errorMsg -match "Insufficient feedback") {
        Write-Host "  ⚠️ Not enough feedback yet. Need at least 10 samples." -ForegroundColor Yellow
        Write-Host "  Submit more feedback and try again." -ForegroundColor Yellow
    } else {
        Write-Host "  ❌ Retraining failed: $errorMsg" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  ✅ WORKFLOW COMPLETE!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next: Check /models endpoint to see registered models" -ForegroundColor Cyan
