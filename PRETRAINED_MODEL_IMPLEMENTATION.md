# Pre-trained Model Implementation Guide

## Overview

This document describes the implementation of the pre-trained HuggingFace model `eakashyap/product-review-sentiment-analyzer` to replace the previous custom DistilBERT training approach. This change was made to avoid overfitting/underfitting issues and leverage a proven, production-ready model.

## Model Information

- **Model**: `eakashyap/product-review-sentiment-analyzer`
- **Base Architecture**: DistilBERT (distilbert-base-uncased)
- **Training Data**: 700k Yelp product reviews
- **Performance**: 90% accuracy on test set
- **Classes**: 3-class sentiment (Negative, Neutral, Positive)
- **Framework**: TensorFlow (converted from PyTorch)

### Label Mapping

The pre-trained model uses the following label mapping:
- **0**: Negative
- **1**: Neutral
- **2**: Positive

This differs from the previous implementation which used:
- **0**: Positive
- **1**: Negative
- **2**: Neutral

All services have been updated to use the new mapping consistently.

## Changes Made

### 1. New SageMaker Scripts

#### a. `train_pretrained.py`
- **Location**: `/sagemaker-scripts/train_pretrained.py`
- **Purpose**: Downloads and saves the pre-trained model for SageMaker deployment
- **Key Features**:
  - Downloads model from HuggingFace Hub
  - Converts from PyTorch to TensorFlow
  - Verifies label mapping with test samples
  - Evaluates on a sample of your dataset (500 examples)
  - Saves model, tokenizer, label mappings, and metrics
  - **No training** by default (epochs=0) - just download and save

#### b. `inference_pretrained.py`
- **Location**: `/sagemaker-scripts/inference_pretrained.py`
- **Purpose**: Handles inference requests for the pre-trained model
- **Key Features**:
  - Loads TensorFlow SavedModel format
  - Loads tokenizer from pickle file
  - Loads reverse label mapping for prediction output
  - Returns predictions in the same format as before

#### c. `requirements_pretrained.txt`
- **Location**: `/sagemaker-scripts/requirements_pretrained.txt`
- **Dependencies**: tensorflow, transformers, torch, pandas, numpy, scikit-learn, sentencepiece, boto3

### 2. Model-Init Service Updates

**File**: `/services/model-init-service/app/main.py`

**Changes**:
1. Updated training script reference:
   - Changed from `train_hf_tf.py` to `train_pretrained.py`
   - Set `epochs: '0'` (no training, just download)

2. Updated inference script reference:
   - Changed from `inference_hf_tf.py` to `inference_pretrained.py`

3. Both `/bootstrap` and `/deploy` endpoints updated

**Impact**:
- Bootstrap process now downloads pre-trained model instead of training from scratch
- Much faster initial setup (~5-10 minutes vs ~15-20 minutes)
- No risk of overfitting on initial deployment

### 3. Retraining Service Updates

**File**: `/services/retraining-service/app/main.py`

**Major Changes**:

1. **Imports**: Added `transformers` library imports
2. **Model Configuration**:
   - Removed custom Keras model architecture
   - Added `MODEL_NAME = "eakashyap/product-review-sentiment-analyzer"`
   - Updated label mappings to match pre-trained model

3. **New Function**: `load_pretrained_model_and_tokenizer()`
   - Downloads pre-trained model from HuggingFace
   - Converts to TensorFlow
   - Returns model and tokenizer ready for fine-tuning

4. **Data Processing**:
   - Replaced Keras Tokenizer with HuggingFace AutoTokenizer
   - Changed from sequences/padding to HuggingFace encoding format
   - Creates TensorFlow datasets with input_ids and attention_mask

5. **Training Configuration**:
   - Lower learning rate for fine-tuning (2e-5 vs 0.001)
   - Fewer epochs (5 vs 10)
   - Smaller batch size (8 vs 32)
   - Same callbacks (EarlyStopping, ReduceLROnPlateau)

6. **Model Saving**:
   - Uses `save_pretrained()` instead of Keras `.save()`
   - Saves reverse label mapping as JSON
   - Compatible with HuggingFace ecosystem

7. **Dependencies**: Updated `requirements.txt` to include transformers, torch, sentencepiece

**Impact**:
- Retraining now fine-tunes the proven pre-trained model
- Better starting point than random initialization
- Faster convergence and better generalization

### 4. Requirements Updates

**Files Updated**:
- `/sagemaker-scripts/requirements_pretrained.txt` (new)
- `/services/retraining-service/requirements.txt` (updated)

**New Dependencies**:
- `transformers>=4.30.0` - HuggingFace transformers library
- `torch` - Required for PyTorch to TensorFlow conversion
- `sentencepiece` - Tokenization library
- `tensorflow>=2.13.0` - Version specification

## Deployment Instructions

### Step 1: Upload SageMaker Scripts to S3

You need to package and upload the new scripts to S3:

```bash
# Navigate to sagemaker-scripts directory
cd sagemaker-scripts

# Create a tar.gz archive of all scripts
tar -czf sourcedir.tar.gz *.py requirements_pretrained.txt

# Upload to S3 (replace with your bucket name)
aws s3 cp sourcedir.tar.gz s3://ml-sentiment-models-<ACCOUNT_ID>/sagemaker-scripts/ --profile vgundu
```

### Step 2: Rebuild Docker Images

The retraining service has updated dependencies, so you need to rebuild:

```bash
# Build retraining service image
docker build -t ml-sentiment-retraining-service:latest ./services/retraining-service

# Tag for ECR (replace with your account ID and region)
docker tag ml-sentiment-retraining-service:latest <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/ml-sentiment-retraining-service:latest

# Push to ECR
aws ecr get-login-password --region us-east-2 --profile vgundu | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com
docker push <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/ml-sentiment-retraining-service:latest
```

### Step 3: Deploy Infrastructure

```bash
cd infrastructure
terraform apply -auto-approve
```

### Step 4: Bootstrap the Model

Once deployed, bootstrap the pre-trained model:

```bash
# Get ALB URL
ALB_URL=$(terraform output -raw alb_url)

# Bootstrap with auto-deploy
curl -X POST "$ALB_URL/model-init/bootstrap?auto_deploy=true"
```

This will:
1. Download the pre-trained model from HuggingFace
2. Evaluate it on a sample of your dataset
3. Save it to S3
4. Automatically deploy to SageMaker endpoint when complete

### Step 5: Test Inference

After the endpoint is deployed (~5-10 minutes):

```bash
# Test prediction
curl -X POST "$ALB_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing! I love it!"}'

# Expected output:
# {"predictions": [{"label": "positive", "confidence": 0.95}]}
```

## How It Works Now

### Initial Deployment Flow

```
1. User calls /model-init/bootstrap?auto_deploy=true
   ↓
2. SageMaker training job starts
   - Runs train_pretrained.py
   - Downloads eakashyap/product-review-sentiment-analyzer from HuggingFace
   - Converts from PyTorch to TensorFlow
   - Evaluates on 500 samples from your dataset
   - Saves to S3
   ↓
3. Auto-deploy thread monitors training
   ↓
4. When complete, creates SageMaker endpoint
   - Uses inference_pretrained.py
   - Loads TensorFlow model
   - Loads tokenizer and label mappings
   ↓
5. Endpoint ready for predictions
```

### Feedback Loop Flow

```
1. User provides feedback via /feedback/submit
   ↓
2. Feedback stored in S3
   ↓
3. When sufficient feedback collected (10+ samples):
   - Call /retrain/retrain
   ↓
4. Retraining service:
   - Downloads pre-trained model
   - Loads feedback from S3
   - Fine-tunes model on feedback data
   - Saves fine-tuned version locally
   - Registers with model registry
   ↓
5. New version available
   (Manual deployment to SageMaker endpoint if desired)
```

## Key Benefits

1. **No Overfitting**: Pre-trained on 700k samples, already well-generalized
2. **Fast Deployment**: No initial training required (~5-10 min vs ~15-20 min)
3. **High Accuracy**: 90% accuracy out of the box
4. **Domain Match**: Trained on product reviews (Yelp), similar to Amazon reviews
5. **Easy Fine-tuning**: Can be fine-tuned later with your specific feedback data
6. **Proven Model**: Widely used, tested, and reliable

## Monitoring and Validation

### Check Training Job Status

```bash
# Get job name from bootstrap response
JOB_NAME="ml-sentiment-training-20241215-123456"

# Check status
curl "$ALB_URL/model-init/status/$JOB_NAME"
```

### Check Endpoint Status

```bash
curl "$ALB_URL/model-init/endpoint-status"
```

### Test Model Performance

```bash
# Test different sentiments
curl -X POST "$ALB_URL/predict" -H "Content-Type: application/json" \
  -d '{"text": "This is terrible, waste of money!"}'
# Expected: negative

curl -X POST "$ALB_URL/predict" -H "Content-Type: application/json" \
  -d '{"text": "It is okay, nothing special."}'
# Expected: neutral

curl -X POST "$ALB_URL/predict" -H "Content-Type: application/json" \
  -d '{"text": "Absolutely love this product!"}'
# Expected: positive
```

## Troubleshooting

### Issue: Training job fails with "No module named 'transformers'"

**Solution**: Ensure `requirements_pretrained.txt` is included in sourcedir.tar.gz:
```bash
cd sagemaker-scripts
tar -czf sourcedir.tar.gz *.py requirements_pretrained.txt
aws s3 cp sourcedir.tar.gz s3://ml-sentiment-models-<ACCOUNT_ID>/sagemaker-scripts/
```

### Issue: Label mappings are incorrect

**Verify label mapping**:
- Check CloudWatch logs for training job
- Look for "Label Mapping Verification" section
- Confirm: 0=Negative, 1=Neutral, 2=Positive

### Issue: Retraining service crashes

**Check dependencies**:
```bash
# SSH into retraining container or check logs
docker logs <retraining-container-id>

# Verify transformers is installed
pip list | grep transformers
```

### Issue: Model predictions are all the same class

**This should NOT happen** with the pre-trained model, but if it does:
1. Check the model evaluation metrics in CloudWatch logs
2. Verify the model downloaded correctly (check file sizes in S3)
3. Try re-running bootstrap

## Migration from Old Model

If you have an existing deployment with the old custom model:

1. **Backup**: The old model artifacts remain in S3
2. **Deploy**: Follow deployment instructions above
3. **Test**: Verify the new model works correctly
4. **Rollback**: If needed, you can manually deploy the old model artifacts

## Next Steps

1. **Collect Feedback**: Start collecting user feedback on predictions
2. **Fine-tune**: When you have 10+ feedback samples, run retraining
3. **Compare**: Compare fine-tuned model performance vs. pre-trained
4. **Iterate**: Continue feedback loop to improve model over time

## Files Changed Summary

### New Files
- `/sagemaker-scripts/train_pretrained.py`
- `/sagemaker-scripts/inference_pretrained.py`
- `/sagemaker-scripts/requirements_pretrained.txt`
- `/PRETRAINED_MODEL_IMPLEMENTATION.md` (this file)

### Modified Files
- `/services/model-init-service/app/main.py`
- `/services/retraining-service/app/main.py`
- `/services/retraining-service/requirements.txt`

### Unchanged (No Impact)
- All other microservices
- Infrastructure configuration
- Dataset
- Feedback mechanism
- Model registry
- API Gateway

## Support

If you encounter issues:
1. Check CloudWatch logs for the relevant service
2. Verify S3 bucket permissions
3. Ensure all scripts are uploaded to S3
4. Check SageMaker IAM role permissions
5. Review this documentation

---

**Last Updated**: 2024-12-15
**Model**: eakashyap/product-review-sentiment-analyzer
**Framework**: TensorFlow 2.13+
**HuggingFace Transformers**: 4.30+
