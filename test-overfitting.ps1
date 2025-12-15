# Test for Overfitting - Use examples from dataset
$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"
$JOB_NAME = "ml-sentiment-training-20251215-143539"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Overfitting Test Script" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Wait for training to complete
Write-Host "Step 1: Waiting for training to complete..." -ForegroundColor Yellow
$maxWait = 1200  # 20 minutes
$elapsed = 0
$completed = $false

while ($elapsed -lt $maxWait -and -not $completed) {
    Start-Sleep -Seconds 60
    $elapsed += 60
    
    try {
        $status = Invoke-RestMethod -Uri "$ALB_URL/model-init/status/$JOB_NAME"
        Write-Host "[$([math]::Floor($elapsed/60))m] Status: $($status.status)" -ForegroundColor Gray
        
        if ($status.status -eq "Completed") {
            Write-Host "✅ Training completed!" -ForegroundColor Green
            $completed = $true
        }
        elseif ($status.status -eq "Failed") {
            Write-Host "❌ Training failed!" -ForegroundColor Red
            exit 1
        }
    }
    catch {
        Write-Host "Error checking status: $_" -ForegroundColor Yellow
    }
}

if (-not $completed) {
    Write-Host "❌ Training timeout after 20 minutes" -ForegroundColor Red
    exit 1
}

# Step 2: Wait for endpoint to be ready
Write-Host ""
Write-Host "Step 2: Waiting for endpoint to be ready..." -ForegroundColor Yellow
$endpointReady = $false
$maxWaitEndpoint = 300  # 5 minutes
$elapsedEndpoint = 0

while ($elapsedEndpoint -lt $maxWaitEndpoint -and -not $endpointReady) {
    Start-Sleep -Seconds 30
    $elapsedEndpoint += 30
    
    try {
        $endpointStatus = Invoke-RestMethod -Uri "$ALB_URL/model-init/endpoint-status"
        Write-Host "[$([math]::Floor($elapsedEndpoint/30)) checks] Endpoint status: $($endpointStatus.status)" -ForegroundColor Gray
        
        if ($endpointStatus.status -eq "InService") {
            Write-Host "✅ Endpoint is ready!" -ForegroundColor Green
            $endpointReady = $true
        }
    }
    catch {
        Write-Host "Error checking endpoint: $_" -ForegroundColor Yellow
    }
}

if (-not $endpointReady) {
    Write-Host "⚠️  Endpoint not ready yet, but continuing with tests..." -ForegroundColor Yellow
}

# Step 3: Test predictions on diverse examples
Write-Host ""
Write-Host "Step 3: Testing predictions on diverse examples..." -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Test cases: Positive, Negative, Neutral, Edge cases
$testCases = @(
    @{text = "Amazon kindle fire has a lot of free app and can be used by any one that wants to get online anywhere,very handy device"; expected = "positive" },
    @{text = "The Echo Show is a great addition to the Amazon family. Works just like the Echo, but with a 7"" screen. Bright vibrant display. Rich clear sound."; expected = "positive" },
    @{text = "nice size, very clear but randomly shuts off. cant remove unwanted apps from the home screen. if you watch video..it buffers alot and laggs."; expected = "negative" },
    @{text = "Needs to be a stand alone device. I should have not required to use a tablet of Cell phone to make it work. Amazon needs to work on the technology on device."; expected = "negative" },
    @{text = "This tablet is on the smaller side but works for me."; expected = "neutral" },
    @{text = "I have enjoyed learning about home automation using Echo plus and consumers need to know that this comes with having to buy additional equipment to get the most out of Alexa."; expected = "neutral" },
    @{text = "I absolutely LOVE my kindle. I go nowhere without it. It has kept me entertained for hours."; expected = "positive" },
    @{text = "Too difficult to setup not compatible with other equipment"; expected = "negative" },
    @{text = "This is a great tablet for the price. I have one that I purchased a few years ago and use it much more than my laptop."; expected = "positive" },
    @{text = "Ended up returning"; expected = "negative" }
)

$results = @()
$correct = 0
$total = $testCases.Count

foreach ($testCase in $testCases) {
    try {
        $body = @{ text = $testCase.text } | ConvertTo-Json
        $response = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body $body -ContentType "application/json"
        
        $predicted = $response.label
        $confidence = $response.confidence
        $isCorrect = ($predicted -eq $testCase.expected)
        
        if ($isCorrect) { $correct++ }
        
        $result = [PSCustomObject]@{
            Text       = $testCase.text.Substring(0, [Math]::Min(60, $testCase.text.Length)) + "..."
            Expected   = $testCase.expected
            Predicted  = $predicted
            Confidence = [math]::Round($confidence, 3)
            Correct    = if ($isCorrect) { "✅" } else { "❌" }
        }
        
        $results += $result
        
        $color = if ($isCorrect) { "Green" } else { "Red" }
        Write-Host "Expected: $($testCase.expected) | Predicted: $predicted (${confidence}) | $($result.Correct)" -ForegroundColor $color
        
    }
    catch {
        Write-Host "Error testing: $($testCase.text.Substring(0, 50))... - $_" -ForegroundColor Red
    }
}

# Step 4: Analyze results
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Test Results Summary" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Accuracy: $correct/$total = $([math]::Round($correct/$total*100, 2))%" -ForegroundColor $(if ($correct / $total -ge 0.7) { "Green" } else { "Yellow" })
Write-Host ""

# Check for overfitting signs
$avgConfidence = ($results | Measure-Object -Property Confidence -Average).Average
$highConfidenceWrong = ($results | Where-Object { $_.Correct -eq "❌" -and $_.Confidence -gt 0.8 }).Count

Write-Host "Average Confidence: $([math]::Round($avgConfidence, 3))" -ForegroundColor Cyan
Write-Host "High Confidence Wrong Predictions (>0.8): $highConfidenceWrong" -ForegroundColor $(if ($highConfidenceWrong -gt 2) { "Red" } else { "Green" })
Write-Host ""

if ($highConfidenceWrong -gt 2) {
    Write-Host "⚠️  WARNING: Potential overfitting detected!" -ForegroundColor Red
    Write-Host "   Model shows high confidence on incorrect predictions." -ForegroundColor Yellow
}
elseif ($correct / $total -lt 0.6) {
    Write-Host "⚠️  WARNING: Low accuracy - model may need improvement" -ForegroundColor Yellow
}
else {
    Write-Host "✅ Model performance looks reasonable" -ForegroundColor Green
}

Write-Host ""
Write-Host "Detailed Results:" -ForegroundColor Cyan
$results | Format-Table -AutoSize
