# Swagger UI Guide - Feedback Loop Workflow

## Accessing Swagger UI

1. Open your browser and go to:
   ```
   http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com/docs
   ```

2. You'll see all available endpoints with interactive testing

---

## Complete Feedback Loop Workflow

### Step 1: Get Prediction

**Endpoint:** `POST /predict`

1. Click on `POST /predict` to expand it
2. Click "Try it out"
3. In the Request body, enter:
   ```json
   {
     "text": "mehh not that great"
   }
   ```
4. Click "Execute"
5. **Copy the response** - you'll need `label` and `confidence` for feedback

**Expected Response:**
```json
{
  "label": "positive",
  "confidence": 0.949166477
}
```

---

### Step 2: Submit Feedback

**Endpoint:** `POST /feedback`

1. Click on `POST /feedback` to expand it
2. Click "Try it out"
3. In the Request body, enter:
   ```json
   {
     "text": "mehh not that great",
     "model_prediction": "positive",
     "user_label": "negative"
   }
   ```
   - `text`: The original text you predicted
   - `model_prediction`: The label from Step 1 (e.g., "positive")
   - `user_label`: The correct label (e.g., "negative" if model was wrong)

4. Click "Execute"

**Expected Response:**
```json
{
  "message": "Feedback submitted",
  "id": 0
}
```

**Important:** You need at least **10 feedback samples** before retraining works!

---

### Step 3: Check Models (Optional)

**Endpoint:** `GET /models`

1. Click on `GET /models`
2. Click "Try it out"
3. Click "Execute"

**Expected Response:**
```json
{
  "models": [
    {
      "version": "sagemaker-ml-sentiment-model-...",
      "path": "s3://...",
      "accuracy": null,
      "is_active": true,
      "source": "sagemaker",
      "endpoint_status": "InService"
    }
  ],
  "count": 1,
  "local_count": 0,
  "sagemaker_count": 1
}
```

---

### Step 4: Run Evaluation (Optional)

**Endpoint:** `POST /evaluate`

**What it does:**
- Evaluates the current model using feedback data
- Calculates accuracy, precision, recall
- Compares model predictions vs user labels
- Helps you decide if retraining is needed

1. Click on `POST /evaluate`
2. Click "Try it out"
3. Click "Execute" (no body needed)

**Expected Response:**
```json
{
  "message": "Evaluation completed",
  "accuracy": 0.85,
  "precision": 0.82,
  "recall": 0.88,
  "f1_score": 0.85,
  "total_samples": 50
}
```

**When to use:**
- After collecting feedback
- Before retraining (to see if model needs improvement)
- To track model performance over time

---

### Step 5: Trigger Retraining

**Endpoint:** `POST /retrain`

**Prerequisites:**
- ✅ At least 10 feedback samples submitted
- ✅ Feedback stored in S3

1. Click on `POST /retrain`
2. Click "Try it out"
3. Click "Execute" (no body needed)

**Expected Response:**
```json
{
  "message": "Model retrained and registered",
  "version": "v20251214123456",
  "path": "/models/model_v20251214123456.keras",
  "accuracy": 0.92,
  "training_samples": 15
}
```

**What happens:**
1. Loads all feedback from S3
2. Trains new model on feedback data
3. Evaluates new model
4. Registers model in model-registry
5. Sends notification

**Note:** This uses the retraining-service which trains locally. For SageMaker retraining, use `/model-init/bootstrap` instead.

---

## Quick Reference

| Endpoint | Purpose | When to Use |
|----------|---------|-------------|
| `POST /predict` | Get sentiment prediction | Every time you want to classify text |
| `POST /feedback` | Submit user correction | When model prediction is wrong |
| `GET /models` | List all models | Check what models are available |
| `POST /evaluate` | Evaluate model performance | After collecting feedback, before retraining |
| `POST /retrain` | Retrain model on feedback | After 10+ feedback samples collected |

---

## Tips

1. **Collect feedback first:** Submit at least 10 feedback samples before retraining
2. **Use evaluation:** Run `/evaluate` to see if retraining is actually needed
3. **Check models:** Use `/models` to see all your models (local + SageMaker)
4. **SageMaker vs Local:** 
   - `/retrain` = Local retraining (fast, uses feedback data)
   - `/model-init/bootstrap` = SageMaker training (slower, uses full dataset)

---

## Troubleshooting

**"Insufficient feedback data" error:**
- You need at least 10 feedback samples
- Submit more feedback using `/feedback` endpoint

**"No active model found":**
- Check `/models` to see available models
- Trigger training using `/model-init/bootstrap`

**Evaluation returns empty:**
- Make sure you've submitted feedback first
- Check that feedback is stored in S3
