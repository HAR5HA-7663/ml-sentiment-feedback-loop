# Quick Swagger UI Guide - Retraining Workflow

## 🚀 Access Swagger UI

Open in browser:
```
http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com/docs
```

---

## 📝 Step-by-Step: Retraining in Swagger UI

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
4. Click "Execute"
5. **Repeat this 10+ times with different texts** to collect enough feedback

**Tip:** Use different texts each time:
- "This is terrible!" → user_label: "negative"
- "Amazing product!" → user_label: "positive"
- "It's okay" → user_label: "neutral"

---

### Step 3: Run Evaluation (Optional but Recommended)

**What `/evaluate` does:**
- Compares model predictions vs user labels
- Calculates accuracy: How many predictions were correct?
- Helps you decide: "Is my model good enough, or does it need retraining?"

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
- If accuracy < 0.80, consider retraining
- If accuracy > 0.90, model is doing well

**Why use it?**
- ✅ See if retraining is actually needed
- ✅ Track model performance over time
- ✅ Make data-driven decisions

---

### Step 4: Trigger Retraining

**Prerequisites:**
- ✅ At least 10 feedback samples submitted
- ✅ (Optional) Run evaluation first

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

---

## 🎯 Why Use `/evaluate`?

### Purpose:
**Evaluate** = Measure current model performance using feedback data

### When to use:
1. **Before retraining:** "Is my model bad enough to retrain?"
2. **After collecting feedback:** "How accurate is my model?"
3. **Track progress:** "Is my model getting better or worse?"

### What it calculates:
- **Accuracy:** % of correct predictions
- **Correct/Total:** How many predictions matched user labels

### Example Workflow:
```
1. Collect 20 feedback samples
2. Run /evaluate → accuracy: 0.60 (60% correct)
3. Decision: "60% is too low, let's retrain"
4. Run /retrain → new model with 85% accuracy
5. Run /evaluate again → verify improvement
```

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

## 📊 Quick Reference

| Endpoint | Purpose | Body Required? |
|----------|---------|----------------|
| `POST /predict` | Get prediction | Yes: `{"text": "..."}` |
| `POST /feedback` | Submit correction | Yes: `{"text": "...", "model_prediction": "...", "user_label": "..."}` |
| `POST /evaluate` | Check model accuracy | No |
| `POST /retrain` | Train new model | No |
| `GET /models` | List all models | No |

---

## 🎓 Best Practices

1. **Collect feedback first:** Submit 10-20 feedback samples
2. **Evaluate before retraining:** See if retraining is needed
3. **Evaluate after retraining:** Verify improvement
4. **Track over time:** Run evaluation regularly to monitor model drift
