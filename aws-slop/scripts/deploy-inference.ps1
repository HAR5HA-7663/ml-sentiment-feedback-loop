# Deploy inference-service directly to ECS

$ECR_REGISTRY = "143519759870.dkr.ecr.us-east-2.amazonaws.com"
$SERVICE = "inference-service"
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
