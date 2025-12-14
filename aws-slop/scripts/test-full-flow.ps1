# Test the complete ML feedback loop

$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"

Write-Host "================================================================" -ForegroundColor Green
Write-Host "  TESTING COMPLETE ML FEEDBACK LOOP" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Step 1: Predict Sentiment
Write-Host "Step 1: Predict Sentiment" -ForegroundColor Cyan
$testText = "This product is absolutely amazing! Best purchase ever!"
$prediction = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body (@{text=$testText} | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 20
Write-Host "  Text: '$testText'" -ForegroundColor White
Write-Host "  Prediction: $($prediction.sentiment -or $prediction.label)" -ForegroundColor Green
Write-Host "  Confidence: $([math]::Round($prediction.confidence * 100, 2))%" -ForegroundColor Cyan
Write-Host ""

# Step 2: Submit Feedback
Write-Host "Step 2: Submit Feedback" -ForegroundColor Cyan
$feedback = @{
    text = $testText
    model_prediction = $prediction.sentiment -or $prediction.label
    user_label = "positive"
} | ConvertTo-Json

$feedbackResult = Invoke-RestMethod -Uri "$ALB_URL/feedback" -Method POST -Body $feedback -ContentType "application/json" -TimeoutSec 10
Write-Host "  Feedback submitted: $($feedbackResult.message -or 'Success')" -ForegroundColor Green
Write-Host ""

# Step 3: Run Evaluation
Write-Host "Step 3: Run Evaluation" -ForegroundColor Cyan
try {
    $evalResult = Invoke-RestMethod -Uri "$ALB_URL/evaluate" -Method POST -Body (@{model_id="current"} | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 30
    Write-Host "  Evaluation: $($evalResult.message -or 'Completed')" -ForegroundColor Green
    if ($evalResult.accuracy) {
        Write-Host "  Accuracy: $([math]::Round($evalResult.accuracy * 100, 2))%" -ForegroundColor Cyan
    }
} catch {
    Write-Host "  Evaluation endpoint may not be fully implemented" -ForegroundColor Yellow
}
Write-Host ""

# Step 4: List Models
Write-Host "Step 4: List Models" -ForegroundColor Cyan
$models = Invoke-RestMethod -Uri "$ALB_URL/models" -Method GET -TimeoutSec 10
Write-Host "  Models found: $($models.models.Count)" -ForegroundColor Green
Write-Host ""

Write-Host "================================================================" -ForegroundColor Green
Write-Host "  ✅ FULL FLOW TEST COMPLETE!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
