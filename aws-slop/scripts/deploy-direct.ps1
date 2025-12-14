# Direct ECS Deployment Script - Fast iteration without GitHub Actions
# This builds and deploys directly to ECS in ~2-3 minutes instead of 20-30

param(
    [string]$Service = "inference-service",  # Service to deploy
    [string]$Region = "us-east-2"
)

$ECR_REGISTRY = "143519759870.dkr.ecr.us-east-2.amazonaws.com"
$CLUSTER = "ml-sentiment-cluster"
$SERVICE_NAME = "ml-sentiment-$Service"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  DIRECT ECS DEPLOYMENT (FAST)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Service: $SERVICE_NAME" -ForegroundColor Yellow
Write-Host "Region: $Region" -ForegroundColor Yellow
Write-Host ""

# Step 1: Build Docker image locally
Write-Host "Step 1: Building Docker image..." -ForegroundColor Cyan
$imageTag = "dev-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$imageName = "$ECR_REGISTRY/ml-sentiment-$Service`:$imageTag"

Set-Location "services/$Service"
docker build -t $imageName .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    exit 1
}
Set-Location ../..

Write-Host "✅ Image built: $imageName" -ForegroundColor Green
Write-Host ""

# Step 2: Login to ECR
Write-Host "Step 2: Logging into ECR..." -ForegroundColor Cyan
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $ECR_REGISTRY
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ ECR login failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✅ ECR login successful" -ForegroundColor Green
Write-Host ""

# Step 3: Push image
Write-Host "Step 3: Pushing image to ECR..." -ForegroundColor Cyan
docker push $imageName
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Image push failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Image pushed" -ForegroundColor Green
Write-Host ""

# Step 4: Update ECS service
Write-Host "Step 4: Updating ECS service..." -ForegroundColor Cyan
aws ecs update-service `
    --cluster $CLUSTER `
    --service $SERVICE_NAME `
    --force-new-deployment `
    --region $Region | Out-Null

Write-Host "✅ Service update initiated" -ForegroundColor Green
Write-Host ""
Write-Host "Waiting for deployment to stabilize (30 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Step 5: Check status
Write-Host ""
Write-Host "Step 5: Checking deployment status..." -ForegroundColor Cyan
$service = aws ecs describe-services --cluster $CLUSTER --services $SERVICE_NAME --region $Region | ConvertFrom-Json
$running = $service.services[0].runningCount
$desired = $service.services[0].desiredCount

Write-Host "Running: $running/$desired" -ForegroundColor $(if ($running -eq $desired) { "Green" } else { "Yellow" })
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  ✅ DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Image: $imageName" -ForegroundColor Cyan
Write-Host ""
Write-Host "Monitor logs:" -ForegroundColor Yellow
Write-Host "  aws logs tail /ecs/ml-sentiment/$Service --follow --region $Region" -ForegroundColor White
Write-Host ""
