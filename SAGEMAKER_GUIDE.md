# SageMaker ML Training Flow - Complete Guide

## 🎯 **Overview**

This guide walks you through triggering SageMaker training and deploying your ML model.

---

## ⏱️ **Timeline**

1. **Infrastructure Deployment**: ~10 minutes (automatic via GitHub Actions)
2. **Upload Scripts to S3**: ~1 minute (automatic via GitHub Actions)
3. **Trigger Training**: Manual (you call API)
4. **SageMaker Training**: ~15-20 minutes (SageMaker handles)
5. **Deploy Endpoint**: Manual (you call API)
6. **Endpoint Ready**: ~3-5 minutes (SageMaker handles)

**Total: ~30-40 minutes**

---

## 📋 **Step-by-Step Instructions**

### **Step 1: Wait for Infrastructure Deployment**

Check GitHub Actions: https://github.com/HAR5HA-7663/ml-sentiment-feedback-loop/actions

Wait until the "CI/CD Pipeline" workflow completes successfully (~10 minutes).

### **Step 2: Get Your ALB URL**

```powershell
# Run this to get your ALB URL
aws elbv2 describe-load-balancers --region us-east-2 --query "LoadBalancers[?contains(LoadBalancerName, 'ml-sentiment')].DNSName" --output text
```

Save this URL, you'll use it for all API calls.

Example: `ml-sentiment-alb-123456789.us-east-2.elb.amazonaws.com`

---

### **Step 3: Start SageMaker Training**

```powershell
# Set your ALB URL
$ALB_URL = "http://ml-sentiment-alb-XXXXXXXXX.us-east-2.elb.amazonaws.com"

# Start training via model-init-service
$response = Invoke-RestMethod -Uri "$ALB_URL/model-init/bootstrap" -Method POST
$response | ConvertTo-Json

# Save the job name for later
$JOB_NAME = $response.training_job_name
echo "Training job: $JOB_NAME"
```

**Expected Output:**
```json
{
  "message": "SageMaker training job started",
  "training_job_name": "ml-sentiment-training-20251213-195432",
  "status": "InProgress",
  "estimated_time": "15-20 minutes",
  "note": "Use /status endpoint to check progress"
}
```

---

### **Step 4: Monitor Training Progress**

```powershell
# Check training status (run this every few minutes)
Invoke-RestMethod -Uri "$ALB_URL/model-init/status/$JOB_NAME" | ConvertTo-Json

# Or watch it continuously
while ($true) {
    $status = Invoke-RestMethod -Uri "$ALB_URL/model-init/status/$JOB_NAME"
    Write-Host "Status: $($status.status) - $(Get-Date)" -ForegroundColor Cyan
    
    if ($status.status -eq "Completed") {
        Write-Host "✅ Training Complete!" -ForegroundColor Green
        break
    }
    elseif ($status.status -eq "Failed") {
        Write-Host "❌ Training Failed: $($status.failure_reason)" -ForegroundColor Red
        break
    }
    
    Start-Sleep -Seconds 60
}
```

**Status Values:**
- `InProgress`: Training is running
- `Completed`: Training finished successfully
- `Failed`: Training encountered an error

---

### **Step 5: View Training in AWS Console**

While waiting, check the AWS SageMaker Console:

1. Go to: https://console.aws.amazon.com/sagemaker
2. Region: **us-east-2 (Ohio)**
3. Left sidebar: **Training** → **Training jobs**
4. Find your job: `ml-sentiment-training-XXXXXXXX`
5. Click it to see:
   - Training metrics
   - CloudWatch logs
   - Resource utilization

**Show this to your professor!** 🎓

---

### **Step 6: Deploy Model to Endpoint**

Once training is complete:

```powershell
# Deploy the trained model to endpoint
Invoke-RestMethod -Uri "$ALB_URL/model-init/deploy/$JOB_NAME" -Method POST | ConvertTo-Json
```

**Expected Output:**
```json
{
  "message": "Model deployment created",
  "endpoint_name": "ml-sentiment-endpoint",
  "model_name": "ml-sentiment-model-20251213-200123",
  "status": "Creating",
  "estimated_time": "3-5 minutes"
}
```

---

### **Step 7: Wait for Endpoint to be Ready**

```powershell
# Check endpoint status
while ($true) {
    $status = Invoke-RestMethod -Uri "$ALB_URL/model-init/endpoint-status"
    Write-Host "Endpoint Status: $($status.status) - $(Get-Date)" -ForegroundColor Cyan
    
    if ($status.status -eq "InService") {
        Write-Host "✅ Endpoint Ready!" -ForegroundColor Green
        break
    }
    elseif ($status.status -eq "Failed") {
        Write-Host "❌ Endpoint Failed!" -ForegroundColor Red
        break
    }
    
    Start-Sleep -Seconds 30
}
```

---

### **Step 8: Test Real ML Predictions**

```powershell
# Test sentiment prediction through API Gateway
$body = @{
    text = "This product is absolutely amazing! I love it!"
} | ConvertTo-Json

Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body $body -ContentType "application/json" | ConvertTo-Json
```

**Expected Output:**
```json
{
  "label": "positive",
  "confidence": 0.9234
}
```

**NOT "neutral" anymore!** Real ML predictions! 🎉

---

### **Step 9: Test Complete Feedback Loop**

```powershell
# 1. Get prediction
$prediction = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body (@{text = "This is great!"} | ConvertTo-Json) -ContentType "application/json"
Write-Host "Prediction: $($prediction.label)" -ForegroundColor Yellow

# 2. Submit feedback (user corrects it)
$feedback = @{
    text = "This is great!"
    model_prediction = $prediction.label
    user_label = "positive"
} | ConvertTo-Json

Invoke-RestMethod -Uri "$ALB_URL/feedback" -Method POST -Body $feedback -ContentType "application/json"
Write-Host "Feedback submitted" -ForegroundColor Green

# 3. Run evaluation
Invoke-RestMethod -Uri "$ALB_URL/evaluate" -Method POST | ConvertTo-Json
```

---

## 🎓 **Demo for Professor**

### **Show These 5 Things:**

1. **AWS SageMaker Console**
   - Show completed training job
   - Show training metrics/logs
   - Show deployed endpoint (InService)

2. **API Gateway**
   - Show unified entry point
   - Show health checks for all services

3. **Real ML Predictions**
   - Show `/predict` returning real sentiment (not neutral!)
   - Show confidence scores

4. **Feedback Collection**
   - Show feedback being submitted
   - Show S3 bucket with feedback data

5. **Model Evaluation**
   - Show `/evaluate` calculating accuracy
   - Show evaluation reading from S3

---

## 🔍 **Troubleshooting**

### **Training Failed?**

Check CloudWatch logs:
```powershell
aws logs tail /aws/sagemaker/TrainingJobs --follow --region us-east-2
```

### **Endpoint Not Ready?**

Check endpoint status in AWS Console:
- SageMaker → Endpoints → ml-sentiment-endpoint

### **Inference Returning Neutral?**

Endpoint might not be ready yet. Check:
```powershell
Invoke-RestMethod -Uri "$ALB_URL/model-init/endpoint-status"
```

---

## 📊 **What You Built**

✅ **Production SageMaker Pipeline**
- Real ML training on AWS infrastructure
- Automated model deployment
- Scalable inference endpoints

✅ **Complete MLOps Workflow**
- Training → Evaluation → Deployment → Monitoring
- Feedback collection → Retraining capability

✅ **Microservices Architecture**
- API Gateway pattern
- Service discovery
- Cloud-native deployment

**This is a real production-grade ML system!** 🚀
