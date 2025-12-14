# Swagger UI Guide - Complete Retraining Workflow

## 🚀 Access Swagger UI

Open in your browser:
```
http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com/docs
```

---

## 📝 Complete Workflow: Prediction → Feedback → Evaluation → Retraining

### Step 1: Get Prediction

1. Find `POST /predict` in Swagger UI
2. Click "Try it out"
3. Enter:
   ```json
   {
     "text": "mehh not that great"
   }
   ```
4. Click "Execute"
5. **Copy the `label` from response** (e.g., "positive")

---

### Step 2: Submit Feedback (Repeat 10+ times!)

**Why 10+?** Retraining requires at least 10 feedback samples.

1. Find `POST /feedback`
2. Click "Try it out"
3. Enter:
   ```json
   {
     "text": "mehh not that great",
     "model_prediction": "positive",
     "user_label": "negative"
   }
   ```
   - `text`: Original text
   - `model_prediction`: Label from Step 1
   - `user_label`: Correct label (your correction)

4. Click "Execute"
5. **Repeat with different texts** to collect 10+ samples

**Example texts to try:**
- "This is terrible!" → `user_label: "negative"`
- "Amazing product!" → `user_label: "positive"`
- "It's okay, nothing special" → `user_label: "neutral"`

---

### Step 3: Run Evaluation (Why? Explained below)

**What `/evaluate` does:**
- Compares model predictions vs user labels
- Calculates **accuracy**: How many predictions were correct?
- Helps you decide: "Should I retrain or is the model good enough?"

1. Find `POST /evaluate`
2. Click "Try it out"
3. Click "Execute" (no body needed)

**Response Example:**
```json
{
  "accuracy": 0.75,
  "correct": 15,
  "total": 20,
  "timestamp": "2025-12-14T12:34:56"
}
```

**Interpretation:**
- `accuracy: 0.75` = 75% correct (25% wrong)
- If accuracy < 0.80 → Consider retraining
- If accuracy > 0.90 → Model is doing well

**Why use `/evaluate`?**
- ✅ **Before retraining:** See if retraining is actually needed
- ✅ **After collecting feedback:** Measure current model performance
- ✅ **Track progress:** Monitor if model is getting better or worse
- ✅ **Data-driven decisions:** Don't retrain blindly - check accuracy first!

**Example Workflow:**
```
1. Collect 20 feedback samples
2. Run /evaluate → accuracy: 0.60 (60% correct)
3. Decision: "60% is too low, let's retrain"
4. Run /retrain → new model with 85% accuracy
5. Run /evaluate again → verify improvement to 85%
```

---

### Step 4: Trigger Retraining

**Prerequisites:**
- ✅ At least 10 feedback samples submitted
- ✅ (Recommended) Run evaluation first to see if retraining is needed

1. Find `POST /retrain`
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
3. Evaluates new model accuracy
4. Registers in model registry
5. Sends notification

---

### Step 5: Check Models

1. Find `GET /models`
2. Click "Try it out"
3. Click "Execute"

**You'll see:**
- Local models (from retraining-service)
- SageMaker models (from deployed endpoint)
- Training jobs

**Response:**
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
    },
    {
      "version": "v20251214123456",
      "path": "/models/model_v20251214123456.keras",
      "accuracy": 0.92,
      "is_active": true,
      "source": "local"
    }
  ],
  "count": 2,
  "local_count": 1,
  "sagemaker_count": 1
}
```

---

## 🎯 Why Use `/evaluate`? (Detailed Explanation)

### Purpose
**Evaluate** = Measure current model performance using feedback data

### What It Calculates
- **Accuracy:** Percentage of predictions that matched user labels
- **Correct/Total:** How many predictions were correct out of total feedback

### When to Use

1. **Before Retraining:**
   - Question: "Is my model bad enough to retrain?"
   - Action: Run `/evaluate`
   - Decision:
     - If accuracy < 0.70 → Definitely retrain
     - If accuracy 0.70-0.85 → Consider retraining
     - If accuracy > 0.90 → Model is good, maybe skip retraining

2. **After Collecting Feedback:**
   - Question: "How accurate is my current model?"
   - Action: Run `/evaluate` to get baseline accuracy
   - Use: Compare before/after retraining

3. **Track Progress Over Time:**
   - Run `/evaluate` regularly (weekly/monthly)
   - Monitor if accuracy is declining (model drift)
   - Trigger retraining when accuracy drops below threshold

### Example Decision Tree

```
Collect 20 feedback samples
    ↓
Run /evaluate
    ↓
Accuracy = 0.60 (60%)
    ↓
Decision: "Too low! Retrain needed"
    ↓
Run /retrain
    ↓
New model accuracy = 0.85 (85%)
    ↓
Run /evaluate again
    ↓
Verify: Accuracy improved from 60% → 85% ✅
```

---

## 📊 Quick Reference

| Endpoint | Purpose | Body Required? | When to Use |
|----------|---------|----------------|-------------|
| `POST /predict` | Get prediction | Yes: `{"text": "..."}` | Every time you want to classify text |
| `POST /feedback` | Submit correction | Yes: `{"text": "...", "model_prediction": "...", "user_label": "..."}` | When model prediction is wrong |
| `POST /evaluate` | Check model accuracy | No | After collecting feedback, before retraining |
| `POST /retrain` | Train new model | No | After 10+ feedback samples collected |
| `GET /models` | List all models | No | Check what models are available |

---

## ⚠️ Common Issues

**"Insufficient feedback data" when retraining:**
- You need at least 10 feedback samples
- Submit more using `/feedback` endpoint

**Evaluation returns empty:**
- Make sure you've submitted feedback first
- Check that feedback is stored in S3

**Models endpoint shows empty:**
- Check if SageMaker endpoint is deployed
- Check if any models are registered

---

## 🎓 Best Practices

1. **Collect feedback first:** Submit 10-20 feedback samples
2. **Evaluate before retraining:** See if retraining is needed (don't retrain blindly!)
3. **Evaluate after retraining:** Verify improvement
4. **Track over time:** Run evaluation regularly to monitor model drift
