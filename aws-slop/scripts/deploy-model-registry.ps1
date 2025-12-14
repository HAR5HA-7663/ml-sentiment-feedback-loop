# Deploy model-registry-service directly to ECS

$ECR_REGISTRY = "143519759870.dkr.ecr.us-east-2.amazonaws.com"
$SERVICE = "model-registry-service"
$CLUSTER = "ml-sentiment-cluster"
$SERVICE_NAME = "ml-sentiment-$SERVICE"
$imageTag = "dev-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$imageName = "$ECR_REGISTRY/ml-sentiment-$SERVICE`:$imageTag"

Write-Host "Building image..." -ForegroundColor Cyan
Set-Location "services/$SERVICE"
docker build -t $imageName .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed" -ForegroundColor Red
    Set-Location ../..
    exit 1
}
Set-Location ../..

Write-Host "Logging into ECR..." -ForegroundColor Cyan
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin $ECR_REGISTRY

Write-Host "Pushing image..." -ForegroundColor Cyan
docker push $imageName

Write-Host "Updating ECS service..." -ForegroundColor Cyan
aws ecs update-service --cluster $CLUSTER --service $SERVICE_NAME --force-new-deployment --region us-east-2 | Out-Null

Write-Host "✅ Deployment initiated!" -ForegroundColor Green
Write-Host "Image: $imageName" -ForegroundColor Cyan
Write-Host ""
Write-Host "Waiting 90 seconds for service to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 90
Write-Host "Testing /models endpoint..." -ForegroundColor Cyan
$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"
try {
    $models = Invoke-RestMethod -Uri "$ALB_URL/models" -Method GET -TimeoutSec 10
    Write-Host "✅ Models endpoint working!" -ForegroundColor Green
    Write-Host "  Total models: $($models.count)" -ForegroundColor Cyan
    Write-Host "  SageMaker models: $($models.sagemaker_count)" -ForegroundColor Cyan
    Write-Host "  Local models: $($models.local_count)" -ForegroundColor Cyan
} catch {
    Write-Host "⚠️ Models endpoint not ready yet: $($_.Exception.Message)" -ForegroundColor Yellow
}
