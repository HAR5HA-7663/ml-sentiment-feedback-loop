# SageMaker Training CLI Guide

## ✅ Current Training Job

**Job Name:** `ml-sentiment-training-20251214-223057`  
**Status:** InProgress  
**Estimated Time:** 15-20 minutes

---

## 📋 Quick Commands

### Check Training Status

```powershell
$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"
$JOB_NAME = "ml-sentiment-training-20251214-223057"
Invoke-RestMethod -Uri "$ALB_URL/model-init/status/$JOB_NAME" | ConvertTo-Json
```

### Monitor Training (Auto-refresh)

```powershell
.\aws-slop\scripts\monitor-training.ps1 -JobName "ml-sentiment-training-20251214-223057"
```

### Deploy Model After Training Completes

```powershell
$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"
$JOB_NAME = "ml-sentiment-training-20251214-223057"
Invoke-RestMethod -Uri "$ALB_URL/model-init/deploy/$JOB_NAME" -Method POST | ConvertTo-Json
```

### Check Endpoint Status

```powershell
$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"
Invoke-RestMethod -Uri "$ALB_URL/model-init/endpoint-status" | ConvertTo-Json
```

---

## 🔄 Start New Training Job

```powershell
$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"
$response = Invoke-RestMethod -Uri "$ALB_URL/model-init/bootstrap" -Method POST
$response | ConvertTo-Json
$JOB_NAME = $response.training_job_name
Write-Host "Job Name: $JOB_NAME"
```

---

## 📊 Training Data Location

**S3 Bucket:** `ml-sentiment-data-143519759870`  
**File:** `train_data.csv`  
**Status:** ✅ Already uploaded

### Upload New Training Data (if needed)

```powershell
aws s3 cp train_data.csv s3://ml-sentiment-data-143519759870/train_data.csv
```

---

## 🎯 Complete Workflow

1. **Start Training:**

   ```powershell
   $ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"
   $response = Invoke-RestMethod -Uri "$ALB_URL/model-init/bootstrap" -Method POST
   $JOB_NAME = $response.training_job_name
   ```

2. **Monitor Training:**

   ```powershell
   .\aws-slop\scripts\monitor-training.ps1 -JobName $JOB_NAME
   ```

3. **Deploy Model:**

   ```powershell
   Invoke-RestMethod -Uri "$ALB_URL/model-init/deploy/$JOB_NAME" -Method POST
   ```

4. **Test Prediction:**
   ```powershell
   $body = @{text = "This is amazing!"} | ConvertTo-Json
   Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body $body -ContentType "application/json"
   ```

---

## 🔍 AWS Console Links

- **SageMaker Training Jobs:** https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/training-jobs
- **SageMaker Endpoints:** https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/endpoints
- **CloudWatch Logs:** https://console.aws.amazon.com/cloudwatch/home?region=us-east-2#logsV2:log-groups

---

## ⚠️ Troubleshooting

**Training fails:**

- Check CloudWatch logs for the training job
- Verify `train_data.csv` exists in S3
- Check SageMaker execution role permissions

**Endpoint deployment fails:**

- Ensure training job completed successfully
- Check model artifacts in S3: `s3://ml-sentiment-models-143519759870/training-output/`

**API returns 503:**

- Services may still be starting up
- Wait 2-3 minutes and retry
