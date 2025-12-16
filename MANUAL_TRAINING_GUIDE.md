# Manual SageMaker Training Guide - Fix Overfitting

## 🎯 Problem

Model still predicts only "positive" even after 1:1 balance. Need more aggressive fixes.

## 🔧 What to Change

### Option 1: More Aggressive Undersampling + Remove Class Weights

Since we're already balancing the dataset, we shouldn't need class weights. The issue might be:

1. Random seed causing inconsistent undersampling
2. Class weights interfering with balanced data
3. Model still seeing too many positive examples

### Option 2: Increase Epochs + Better Early Stopping

The model might need more training to learn minority classes properly.

---

## 📋 Step-by-Step: Manual Training in SageMaker Console

### Step 1: Prepare Training Script

1. Open: `sagemaker-scripts/train.py`
2. Make these changes (see below)
3. Upload to S3:
   ```powershell
   aws s3 cp sagemaker-scripts/train.py s3://ml-sentiment-models-143519759870/sagemaker-scripts/train.py
   ```

### Step 2: Create Training Job in SageMaker Console

1. **Go to SageMaker Console:**

   - URL: https://console.aws.amazon.com/sagemaker
   - Region: **us-east-2 (Ohio)**

2. **Navigate:**

   - Left menu: **Training** → **Training jobs**
   - Click **Create training job**

3. **Fill in the form:**

   **Basic Information:**

   - Training job name: `ml-sentiment-manual-training-YYYYMMDD-HHMMSS` (use current date/time)
   - IAM role: `ml-sentiment-sagemaker-role` (or select from dropdown)

   **Algorithm:**

   - Select: **Use a custom algorithm**
   - Container image: `763104351884.dkr.ecr.us-east-2.amazonaws.com/tensorflow-training:2.11-cpu-py39`

   **Hyperparameters:**

   ```
   epochs: 10
   batch-size: 32
   sagemaker_program: train.py
   sagemaker_submit_directory: s3://ml-sentiment-models-143519759870/sagemaker-scripts/sourcedir.tar.gz
   sagemaker_region: us-east-2
   ```

   **Input Data:**

   - Channel name: `train`
   - Data source: S3
   - S3 location: `s3://ml-sentiment-data-143519759870/`
   - Content type: `text/csv`
   - Compression: None

   **Output Data:**

   - S3 output path: `s3://ml-sentiment-models-143519759870/training-output/`

   **Resource Configuration:**

   - Instance type: `ml.m5.large`
   - Instance count: `1`
   - Volume size: `30 GB`

   **Environment Variables:**

   ```
   MODELS_BUCKET: ml-sentiment-models-143519759870
   S3_MODELS_BUCKET: ml-sentiment-models-143519759870
   ```

4. **Click "Create training job"**

5. **Wait 15-20 minutes** for training to complete

---

## 🔧 Recommended Code Changes

### Change 1: Fix Random Seed for Consistent Undersampling

Add at the top of `load_and_preprocess_data()`:

```python
np.random.seed(42)  # Fixed seed for reproducible undersampling
```

### Change 2: Remove Class Weights (Already Balanced)

In the `model.fit()` call, remove `class_weight=class_weights`:

```python
history = model.fit(
    X_train_split, y_train_split,
    epochs=args.epochs,
    batch_size=args.batch_size,
    validation_data=(X_val, y_val),
    # class_weight=class_weights,  # REMOVE THIS - already balanced
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
```

### Change 3: Increase Epochs

Change default epochs from 5 to 10:

```python
parser.add_argument('--epochs', type=int, default=10)  # Changed from 5
```

### Change 4: More Aggressive Focal Loss Alpha

In `build_model()`, increase alpha for minority classes:

```python
alpha = tf.constant([0.05, 0.7, 0.25], dtype=tf.float32)  # Even lower for Positive
```

---

## 📊 After Training Completes

### Step 3: Deploy Model

**Option A: Via API (if services are running)**

```powershell
$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"
$JOB_NAME = "ml-sentiment-manual-training-YYYYMMDD-HHMMSS"
Invoke-RestMethod -Uri "$ALB_URL/model-init/deploy/$JOB_NAME" -Method POST
```

**Option B: Via SageMaker Console**

1. Go to **Models** → **Create model**
2. Model name: `ml-sentiment-model-manual-YYYYMMDD-HHMMSS`
3. IAM role: `ml-sentiment-sagemaker-role`
4. Container: Use same image as training
5. Model artifacts: `s3://ml-sentiment-models-143519759870/training-output/ml-sentiment-manual-training-YYYYMMDD-HHMMSS/output/model.tar.gz`
6. Environment variables: Same as training
7. Create endpoint configuration
8. Create/update endpoint

---

## ✅ Test After Deployment

```powershell
$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"

# Test negative
$test1 = @{ text = "nice size, very clear but randomly shuts off" } | ConvertTo-Json
Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body $test1 -ContentType "application/json"

# Test neutral
$test2 = @{ text = "This tablet is on the smaller side but works for me" } | ConvertTo-Json
Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body $test2 -ContentType "application/json"
```

**Expected:** Should see `negative` and `neutral` predictions, not just `positive`.

---

## 🚨 If Still Overfitting

Try these more aggressive fixes:

1. **Oversample minority classes** instead of undersampling
2. **Use SMOTE** (Synthetic Minority Oversampling)
3. **Change model architecture** - simpler model with more regularization
4. **Use different loss function** - weighted cross-entropy instead of focal loss
5. **Check if tokenizer is the issue** - maybe retrain tokenizer on balanced data
