# Quick Fix for Overfitting - Manual Training

## 🎯 The Problem

Model predicts only "positive" even with balanced dataset. Need to fix training script.

## ✅ Quick Fixes to Apply

### 1. Add Random Seed (Line ~38)

```python
# Add this right after imports in load_and_preprocess_data()
np.random.seed(42)  # Fixed seed for reproducible undersampling
```

### 2. Remove Class Weights (Line ~232)

Since dataset is already balanced, remove class weights:

```python
history = model.fit(
    X_train_split, y_train_split,
    epochs=args.epochs,
    batch_size=args.batch_size,
    validation_data=(X_val, y_val),
    # class_weight=class_weights,  # REMOVE THIS LINE
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
```

### 3. Increase Epochs (Line ~180)

```python
parser.add_argument('--epochs', type=int, default=10)  # Change from 5 to 10
```

### 4. More Aggressive Focal Loss (Line ~150)

```python
alpha = tf.constant([0.05, 0.7, 0.25], dtype=tf.float32)  # Lower for Positive
```

---

## 📋 Manual Training Steps

### Step 1: Update Script

Make the 4 changes above to `sagemaker-scripts/train.py`

### Step 2: Upload to S3

```powershell
aws s3 cp sagemaker-scripts/train.py s3://ml-sentiment-models-143519759870/sagemaker-scripts/train.py
```

### Step 3: Create Training Job in Console

**Go to:** https://console.aws.amazon.com/sagemaker (Region: us-east-2)

**Training Job Settings:**

| Setting                        | Value                                                                            |
| ------------------------------ | -------------------------------------------------------------------------------- |
| **Name**                       | `ml-sentiment-manual-YYYYMMDD-HHMMSS`                                            |
| **IAM Role**                   | `ml-sentiment-sagemaker-role`                                                    |
| **Algorithm**                  | Custom algorithm                                                                 |
| **Container Image**            | `763104351884.dkr.ecr.us-east-2.amazonaws.com/tensorflow-training:2.11-cpu-py39` |
| **Hyperparameters**            |                                                                                  |
| - `epochs`                     | `10`                                                                             |
| - `batch-size`                 | `32`                                                                             |
| - `sagemaker_program`          | `train.py`                                                                       |
| - `sagemaker_submit_directory` | `s3://ml-sentiment-models-143519759870/sagemaker-scripts/sourcedir.tar.gz`       |
| - `sagemaker_region`           | `us-east-2`                                                                      |
| **Input Data**                 |                                                                                  |
| - Channel: `train`             | `s3://ml-sentiment-data-143519759870/`                                           |
| - Content Type:                | `text/csv`                                                                       |
| **Output**                     | `s3://ml-sentiment-models-143519759870/training-output/`                         |
| **Instance**                   | `ml.m5.large` (1 instance, 30GB)                                                 |
| **Environment**                |                                                                                  |
| - `MODELS_BUCKET`              | `ml-sentiment-models-143519759870`                                               |
| - `S3_MODELS_BUCKET`           | `ml-sentiment-models-143519759870`                                               |

### Step 4: Wait & Deploy

- Wait 15-20 minutes for training
- Deploy via API or console (see MANUAL_TRAINING_GUIDE.md)

---

## 🧪 Test After Deployment

```powershell
$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"

# Should predict NEGATIVE
$test = @{ text = "nice size, very clear but randomly shuts off" } | ConvertTo-Json
Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body $test -ContentType "application/json"
```

**Expected:** Should return `"label": "negative"`, not `"positive"`
