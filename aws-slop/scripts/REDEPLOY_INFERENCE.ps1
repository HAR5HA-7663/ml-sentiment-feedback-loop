# Quick script to rebuild and redeploy inference service with Hugging Face support
$REGION = "us-east-2"
$ECR_REPO = "ml-sentiment-inference-service"
$SERVICE_NAME = "ml-sentiment-inference-service"
$CLUSTER = "ml-sentiment-cluster"
$ACCOUNT_ID = "143519759870"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "REBUILDING INFERENCE SERVICE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Build and push Docker image
Write-Host "Step 1: Building Docker image..." -ForegroundColor Yellow
cd services/inference-service

$imageTag = "hf-fix-$(Get-Date -Format 'yyyyMMddHHmmss')"
$ecrUri = "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO`:$imageTag"

docker build -t $ecrUri .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Image built: $ecrUri" -ForegroundColor Green
Write-Host ""

# Step 2: Push to ECR
Write-Host "Step 2: Pushing to ECR..." -ForegroundColor Yellow
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
docker push $ecrUri

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker push failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Image pushed!" -ForegroundColor Green
Write-Host ""

# Step 3: Update ECS service
Write-Host "Step 3: Updating ECS service..." -ForegroundColor Yellow
aws ecs update-service `
    --cluster $CLUSTER `
    --service $SERVICE_NAME `
    --force-new-deployment `
    --region $REGION | Out-Null

Write-Host "✅ Service update triggered!" -ForegroundColor Green
Write-Host ""
Write-Host "Waiting for service to stabilize (2-3 minutes)..." -ForegroundColor Yellow

# Wait for service to be stable
$maxWait = 300
$elapsed = 0
$interval = 10

while ($elapsed -lt $maxWait) {
    Start-Sleep -Seconds $interval
    $elapsed += $interval
    
    $service = aws ecs describe-services --cluster $CLUSTER --services $SERVICE_NAME --region $REGION --query 'services[0]' | ConvertFrom-Json
    $running = $service.runningCount
    $desired = $service.desiredCount
    $status = $service.deployments[0].status
    
    Write-Host "[$([math]::Floor($elapsed/60))m] Running: $running/$desired, Status: $status" -ForegroundColor $(if ($status -eq "PRIMARY" -and $running -eq $desired) { "Green" } else { "Yellow" })
    
    if ($status -eq "PRIMARY" -and $running -eq $desired -and $service.deployments.Count -eq 1) {
        Write-Host "`n✅ Service deployed and stable!" -ForegroundColor Green
        break
    }
}

cd ../..
Write-Host "`n✅ Inference service updated with Hugging Face support!" -ForegroundColor Green
Write-Host "Test the endpoint now with: .\test-overfitting.ps1" -ForegroundColor Cyan
