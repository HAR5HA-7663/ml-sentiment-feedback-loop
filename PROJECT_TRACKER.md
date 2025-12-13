# ML Sentiment Feedback Loop - Project Tracker
## SageMaker Pipeline-Driven MLOps System

## Project Overview
Complete ML feedback loop system for product review sentiment analysis using **AWS SageMaker Pipelines** as the execution engine. Microservices serve as the control plane, orchestrating SageMaker training, evaluation, deployment, and inference.

## Architecture Overview

### Microservices (Control Plane)
- **inference-service** → Calls SageMaker Endpoint for predictions
- **feedback-service** → Stores feedback in DynamoDB
- **model-registry-service** → Syncs with SageMaker Model Registry
- **evaluation-service** → Triggers SageMaker Processing Jobs
- **retraining-service** → Triggers SageMaker Pipelines for retraining
- **notification-service** → Publishes to SNS
- **model-init-service** → Bootstraps first model via SageMaker Pipeline

### AWS Components (Execution Engine)
- **SageMaker Pipelines** → Orchestrates training → evaluation → registration → deployment
- **SageMaker Training Jobs** → Model training on GPU instances
- **SageMaker Processing Jobs** → Feature extraction & evaluation
- **SageMaker Model Registry** → Formal model versioning
- **SageMaker Endpoints** → Real-time inference (autoscaling)
- **S3** → Store models, tokenizers, datasets, artifacts
- **DynamoDB** → Store feedback, registry metadata
- **SNS** → Notifications
- **CloudWatch** → Monitoring & logging

## Current Implementation Status

### ✅ Completed (Local Prototype)
- [x] **inference-service** (port 8000)
  - Rule-based sentiment classifier
  - GET /health, POST /predict-sentiment
  - Status: Working, needs SageMaker endpoint integration

- [x] **feedback-service** (port 8001)
  - POST /submit-feedback
  - Stores feedback in feedback.json
  - Status: Working, needs DynamoDB integration

- [x] **model-registry-service** (port 8002)
  - POST /register, GET /models, GET /active-model
  - Stores models.json
  - Status: Working, needs SageMaker Model Registry sync

- [x] **evaluation-service** (port 8003)
  - POST /run-evaluation, GET /latest-evaluation
  - Computes accuracy from feedback
  - Status: Working, needs SageMaker Processing Job

- [x] **retraining-service** (port 8004)
  - POST /retrain
  - Status: Working, needs SageMaker Pipeline trigger

- [x] **notification-service** (port 8005)
  - POST /notify
  - Stores notifications.json
  - Status: Working, needs SNS integration

### 🚧 To Be Implemented

## Implementation Plan

### Phase 1: SageMaker Pipeline Infrastructure

**File**: `infrastructure/sagemaker/pipeline_definition.py`
- Define SageMaker Pipeline with all steps
- Preprocessing step (ProcessingStep)
- Training step (TrainingStep)
- Evaluation step (ProcessingStep)
- Condition step (if accuracy > threshold)
- Register model step
- Deploy endpoint step
- Pipeline parameters (data path, instance types, etc.)

**File**: `infrastructure/sagemaker/preprocessing_script.py`
- Read CSV from S3
- Text tokenization and preprocessing
- Sequence padding
- Train/validation/test split
- Save processed data to S3
- Save tokenizer to S3

**File**: `infrastructure/sagemaker/training_script.py`
- Load preprocessed data from S3
- Build TensorFlow/Keras model (Embedding + LSTM/Dense + Output)
- Train model
- Save SavedModel to S3
- Log metrics to CloudWatch

**File**: `infrastructure/sagemaker/evaluation_script.py`
- Load trained model from S3
- Load test data from S3
- Compute accuracy, confusion matrix
- Generate evaluation report
- Save metrics to S3

**File**: `infrastructure/sagemaker/requirements.txt`
- tensorflow
- pandas
- numpy
- sagemaker

### Phase 2: Model Initialization Service

**File**: `services/model-init-service/app/main.py`
- POST /bootstrap endpoint
- Read Dataset/train_data.csv
- Upload to S3 (s3://bucket/data/train_data.csv)
- Create/trigger SageMaker Pipeline
- Monitor pipeline execution
- Get endpoint name from pipeline output
- Register model in model-registry-service
- Return endpoint name and model version
- Requirements: boto3, sagemaker, pandas

**File**: `services/model-init-service/Dockerfile`
- Python 3.11-slim base
- Install requirements
- Copy app folder

**File**: `services/model-init-service/requirements.txt`
- fastapi
- uvicorn[standard]
- boto3
- sagemaker
- pandas

### Phase 3: Inference Service Update

**File**: `services/inference-service/app/main.py`
- Remove rule-based classifier
- Add SageMaker Runtime client
- Load tokenizer from S3 (or cache locally)
- POST /predict-sentiment endpoint:
  - Preprocess input text
  - Call SageMaker endpoint using invoke_endpoint()
  - Parse response (label + confidence)
  - Return SentimentResponse
- Environment variables: SAGEMAKER_ENDPOINT_NAME, AWS_REGION, S3_BUCKET

**File**: `services/inference-service/requirements.txt`
- Add: boto3, sagemaker

### Phase 4: Feedback Service Update

**File**: `services/feedback-service/app/main.py`
- Add DynamoDB client
- POST /submit-feedback endpoint:
  - Store feedback in DynamoDB table
  - Keep local JSON as fallback for development
- DynamoDB schema:
  - Partition key: feedback_id (UUID)
  - Attributes: text, model_prediction, user_label, timestamp
- Environment variables: DYNAMODB_TABLE_NAME, AWS_REGION

**File**: `services/feedback-service/requirements.txt`
- Add: boto3

### Phase 5: Evaluation Service Update

**File**: `services/evaluation-service/app/main.py`
- Read feedback from DynamoDB
- POST /run-evaluation endpoint:
  - Export feedback to S3 as CSV
  - Trigger SageMaker Processing Job
  - Wait for job completion
  - Read evaluation results from S3
  - Store metrics in DynamoDB or local JSON
- GET /latest-evaluation endpoint:
  - Return last evaluation results
- Environment variables: S3_BUCKET, AWS_REGION, SAGEMAKER_ROLE

**File**: `services/evaluation-service/requirements.txt`
- Add: boto3, sagemaker

### Phase 6: Retraining Service Update

**File**: `services/retraining-service/app/main.py`
- Read feedback from DynamoDB
- POST /retrain endpoint:
  - Merge feedback with historical training data
  - Upload merged dataset to S3
  - Trigger SageMaker Pipeline with new data path
  - Monitor pipeline execution
  - Get new endpoint name
  - Notify notification-service on completion
- Environment variables: S3_BUCKET, AWS_REGION, SAGEMAKER_ROLE

**File**: `services/retraining-service/requirements.txt`
- Add: boto3, sagemaker, httpx

### Phase 7: Model Registry Service Update

**File**: `services/model-registry-service/app/main.py`
- Sync with SageMaker Model Registry
- GET /models endpoint:
  - List models from SageMaker Model Registry
  - Return model versions with metadata
- GET /active-model endpoint:
  - Get active model from SageMaker endpoint
  - Return model details
- POST /register endpoint:
  - Register model in SageMaker Model Registry
  - Keep local JSON as cache
- Environment variables: AWS_REGION, SAGEMAKER_ROLE

**File**: `services/model-registry-service/requirements.txt`
- Add: boto3, sagemaker

### Phase 8: Notification Service Update

**File**: `services/notification-service/app/main.py`
- Add SNS client
- POST /notify endpoint:
  - Publish notification to SNS topic
  - Store in local JSON as fallback
- SNS topic: model-training-notifications
- Environment variables: SNS_TOPIC_ARN, AWS_REGION

**File**: `services/notification-service/requirements.txt`
- Add: boto3

### Phase 9: Docker Compose Updates

**File**: `docker-compose.yml`
- Add model-init-service (port 8006)
- Add environment variables for all services:
  - AWS_REGION
  - AWS_ACCESS_KEY_ID (for local dev)
  - AWS_SECRET_ACCESS_KEY (for local dev)
  - S3_BUCKET
  - DYNAMODB_TABLE_NAME
  - SNS_TOPIC_ARN
  - SAGEMAKER_ROLE
- Add bind mounts for Dataset folder (model-init-service)
- Keep local volumes for development fallback

### Phase 10: AWS Infrastructure Setup

**File**: `infrastructure/aws_setup.md`
- S3 bucket creation:
  - {project-name}-models
  - {project-name}-data
  - {project-name}-artifacts
  - {project-name}-tokenizers
- DynamoDB table creation:
  - feedback-table (Partition key: feedback_id)
  - evaluations-table (Partition key: evaluation_id)
- SNS topic creation:
  - model-training-notifications
- IAM roles:
  - SageMaker execution role (for training/processing jobs)
  - ECS task execution role (if using ECS)
- Security groups and VPC (if needed)

### Phase 11: Documentation

**File**: `README.md`
- System architecture overview
- SageMaker Pipeline explanation
- Quick start guide (local development)
- AWS setup instructions
- IAM roles and permissions
- S3 bucket structure
- API endpoints documentation
- Example workflow: bootstrap → predict → feedback → retrain
- Cost estimation

**File**: `DEPLOYMENT.md`
- Local development setup
- AWS deployment guide
- Environment variables reference
- Troubleshooting guide
- SageMaker Pipeline monitoring

## SageMaker Pipeline Steps

1. **Preprocessing Step**
   - Input: Raw CSV from S3
   - Processing: Tokenization, padding, splitting
   - Output: Processed data + tokenizer to S3

2. **Training Step**
   - Input: Processed data from S3
   - Instance: ml.g4dn.xlarge (GPU)
   - Framework: TensorFlow/Keras
   - Output: SavedModel to S3

3. **Evaluation Step**
   - Input: Trained model + test data
   - Processing: Compute accuracy, confusion matrix
   - Output: Metrics to S3

4. **Condition Step**
   - If accuracy > previous_accuracy → proceed
   - Else → skip deployment

5. **Register Model Step**
   - Register in SageMaker Model Registry
   - Add model metadata

6. **Deploy Endpoint Step**
   - Create/update SageMaker endpoint
   - Blue/Green deployment
   - Return endpoint name

## Data Flow

1. **Bootstrap**: model-init-service → Upload data → Trigger pipeline → Deploy endpoint
2. **Inference**: inference-service → SageMaker Endpoint → Prediction
3. **Feedback**: User → feedback-service → DynamoDB
4. **Evaluation**: evaluation-service → SageMaker Processing Job → Metrics
5. **Retraining**: retraining-service → Trigger pipeline → New model
6. **Notifications**: Pipeline events → notification-service → SNS

## Model Storage
- S3: `s3://bucket/models/model_v{version}/` (SavedModel format)
- S3: `s3://bucket/tokenizers/tokenizer_v{version}.pkl`
- SageMaker Model Registry: Full model metadata and versioning

## AWS Resources Required

### S3 Buckets
- `{project-name}-models` - Model artifacts
- `{project-name}-data` - Training data
- `{project-name}-artifacts` - Pipeline artifacts
- `{project-name}-tokenizers` - Tokenizer files

### DynamoDB Tables
- `feedback-table` - User feedback
- `evaluations-table` - Evaluation results
- `notifications-table` - System notifications (optional, can use SNS)

### SNS Topics
- `model-training-notifications` - Training job updates
- `model-deployment-notifications` - Deployment updates

### SageMaker Resources
- Training jobs (on-demand)
- Processing jobs (on-demand)
- Endpoints (always-on, can use serverless)
- Model Registry (always-on)

### IAM Roles
- SageMaker execution role
- ECS task role (if using ECS)
- Lambda execution role (if using Lambda)

## Cost Estimation (Approximate)

- **SageMaker Training**: ~$1-5 per training job (GPU instance)
- **SageMaker Endpoint**: ~$0.10-0.50/hour (depending on instance)
- **S3 Storage**: ~$0.023/GB/month
- **DynamoDB**: ~$0.25/million reads, $1.25/million writes
- **SNS**: ~$0.50/million requests
- **Total Monthly**: ~$50-200 (depending on usage)

## File Structure (Final)

```
ml-sentiment-feedback-loop/
├── Dataset/
│   ├── train_data.csv
│   ├── test_data.csv
│   └── test_data_hidden.csv
├── infrastructure/
│   ├── sagemaker/
│   │   ├── pipeline_definition.py
│   │   ├── preprocessing_script.py
│   │   ├── training_script.py
│   │   ├── evaluation_script.py
│   │   └── requirements.txt
│   └── aws_setup.md
├── services/
│   ├── inference-service/
│   ├── feedback-service/
│   ├── model-registry-service/
│   ├── evaluation-service/
│   ├── retraining-service/
│   ├── notification-service/
│   └── model-init-service/
├── docker-compose.yml
├── README.md
├── DEPLOYMENT.md
└── PROJECT_TRACKER.md (this file)
```

## Implementation Priority

1. **Week 1**: SageMaker Pipeline infrastructure + model-init-service
2. **Week 2**: Update inference, feedback, evaluation services
3. **Week 3**: Update retraining, registry, notification services
4. **Week 4**: Testing, integration, documentation

## Next Immediate Steps

1. Set up AWS account and credentials
2. Create S3 buckets
3. Create SageMaker Pipeline definition
4. Implement model-init-service
5. Test pipeline execution
6. Update inference-service to use SageMaker endpoint

## Notes

- All microservices remain as FastAPI services
- SageMaker handles all ML operations
- Microservices orchestrate and coordinate
- Can run locally with AWS credentials (for development)
- Full deployment uses ECS/EKS for microservices
- SageMaker endpoints are always in AWS
- Local JSON files serve as fallback for development/testing

