# Using Hugging Face Pretrained Models

## 🎯 Benefits

1. **Better Performance**: Pretrained models (DistilBERT, BERT) understand language better
2. **Less Overfitting**: Pretrained weights reduce overfitting on small datasets
3. **Faster Training**: Fine-tuning is faster than training from scratch
4. **Better Generalization**: Works better on diverse text inputs

## 📋 Setup Steps

### Step 1: Update Requirements

The new training script uses Hugging Face transformers. You need to update the SageMaker container or use a Hugging Face DLC.

**Option A: Use Hugging Face Deep Learning Container (Recommended)**

Change the training image in your SageMaker job:

```
763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04
```

**Option B: Use TensorFlow Container + Install Transformers**

Keep TensorFlow container but install transformers in requirements.txt

### Step 2: Upload Files to S3

```powershell
# Upload new training script
aws s3 cp sagemaker-scripts/train_hf.py s3://ml-sentiment-models-143519759870/sagemaker-scripts/train_hf.py --region us-east-2

# Upload new requirements
aws s3 cp sagemaker-scripts/requirements_hf.txt s3://ml-sentiment-models-143519759870/sagemaker-scripts/requirements.txt --region us-east-2

# Upload new inference script
aws s3 cp sagemaker-scripts/inference_hf.py s3://ml-sentiment-models-143519759870/sagemaker-scripts/inference.py --region us-east-2
```

### Step 3: Create Training Job with Hugging Face

**Using AWS CLI:**

```powershell
$JOB_NAME = "ml-sentiment-hf-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$json = @"
{
  "TrainingJobName": "$JOB_NAME",
  "RoleArn": "arn:aws:iam::143519759870:role/ml-sentiment-sagemaker-role",
  "AlgorithmSpecification": {
    "TrainingImage": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04",
    "TrainingInputMode": "File"
  },
  "InputDataConfig": [
    {
      "ChannelName": "train",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "s3://ml-sentiment-data-143519759870/",
          "S3DataDistributionType": "FullyReplicated"
        }
      },
      "ContentType": "text/csv"
    }
  ],
  "OutputDataConfig": {
    "S3OutputPath": "s3://ml-sentiment-models-143519759870/training-output/"
  },
  "ResourceConfig": {
    "InstanceType": "ml.g4dn.xlarge",
    "InstanceCount": 1,
    "VolumeSizeInGB": 30
  },
  "StoppingCondition": {
    "MaxRuntimeInSeconds": 3600
  },
  "HyperParameters": {
    "epochs": "3",
    "batch-size": "16",
    "learning-rate": "2e-5",
    "sagemaker_program": "train_hf.py",
    "sagemaker_submit_directory": "s3://ml-sentiment-models-143519759870/sagemaker-scripts/sourcedir.tar.gz",
    "sagemaker_region": "us-east-2"
  },
  "Environment": {
    "MODELS_BUCKET": "ml-sentiment-models-143519759870",
    "S3_MODELS_BUCKET": "ml-sentiment-models-143519759870"
  }
}
"@

$tempFile = "$env:TEMP\training-hf.json"
$json | Out-File -FilePath $tempFile -Encoding utf8
aws sagemaker create-training-job --cli-input-json "file://$tempFile" --region us-east-2
Remove-Item $tempFile
```

**Note**: Using GPU instance (`ml.g4dn.xlarge`) for faster training. If budget is tight, use `ml.m5.large` (CPU) but training will be slower.

### Step 4: Deploy Model

After training completes, deploy using the same process but update the inference image:

```powershell
# Use Hugging Face inference container
Image: 763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04
```

## 🔄 Alternative: Use CPU Instance (Cheaper)

If GPU is too expensive, use CPU instance:

```powershell
"InstanceType": "ml.m5.large"  # CPU instance
```

And use CPU-optimized container:

```
763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04
```

## 📊 Expected Improvements

- **Better accuracy** on minority classes (negative/neutral)
- **Less overfitting** - model won't predict only "positive"
- **Faster convergence** - fine-tuning takes 3-5 epochs vs 10+
- **Better generalization** - works on diverse text inputs

## 🧪 Test After Deployment

```powershell
$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"

# Should now predict negative correctly
$test = @{ text = "nice size, very clear but randomly shuts off" } | ConvertTo-Json
Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body $test -ContentType "application/json"
```

## 💰 Cost Considerations

- **GPU (ml.g4dn.xlarge)**: ~$0.70/hour - Faster training (~10-15 min)
- **CPU (ml.m5.large)**: ~$0.10/hour - Slower training (~30-45 min)

For your $180/month budget, CPU is safer but GPU is faster for iteration.
