# ML Sentiment Feedback Loop - MLOps System

A production-ready microservices-based MLOps system for sentiment analysis with automated model retraining, evaluation, and deployment using AWS SageMaker.

## 📋 Table of Contents

- [Problem Statement and Scope](#problem-statement-and-scope)
- [High-Level Architecture](#high-level-architecture)
- [Service Responsibilities](#service-responsibilities)
- [API Specifications](#api-specifications)
- [CI/CD Workflow](#cicd-workflow)
- [System Overview](#system-overview)
- [Setup Instructions](#setup-instructions)
- [Default Accounts and Test Data](#default-accounts-and-test-data)
- [Repository Links](#repository-links)

---

## Problem Statement and Scope

### Problem Statement

Traditional ML model deployment pipelines require manual intervention at multiple stages:
- Manual triggering of retraining when new data arrives
- Manual model evaluation and validation
- Manual deployment of new models to production
- Lack of feedback loop integration from production predictions

This project addresses these challenges by implementing an **automated ML feedback loop** that:
- Continuously collects user feedback on predictions
- Automatically triggers retraining when sufficient feedback is available
- Evaluates model performance and deploys improved models automatically
- Maintains model versioning and rollback capabilities

### Scope

**In Scope:**
- ✅ Sentiment analysis for product reviews (Positive/Negative/Neutral)
- ✅ Real-time inference via SageMaker endpoints
- ✅ Automated feedback collection and storage
- ✅ Model retraining with feedback data
- ✅ Model evaluation and metrics tracking
- ✅ Automated deployment pipeline
- ✅ Model registry and versioning
- ✅ CI/CD integration with GitHub Actions
- ✅ Infrastructure as Code (Terraform)
- ✅ Auto-deployment feature (training → deployment automation)

**Out of Scope:**
- Multi-tenant support
- Real-time streaming inference
- Model explainability features
- A/B testing framework

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Gateway                              │
│              (Single Entry Point - Port 8080)                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Inference   │   │   Feedback   │   │   Model     │
│   Service    │   │   Service    │   │  Registry   │
│   (8000)     │   │   (8001)     │   │   (8002)    │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                   │
       │                  ▼                   │
       │          ┌──────────────┐            │
       │          │     S3       │            │
       │          │  (Feedback)   │            │
       │          └──────────────┘            │
       │                  │                   │
       ▼                  ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Evaluation  │   │  Retraining  │   │  Model Init  │
│   Service    │   │   Service    │   │   Service    │
│   (8003)     │   │   (8004)     │   │   (8006)     │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                   │
       │                  │                   │
       └──────────────────┼───────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   AWS SageMaker       │
              │  - Training Jobs      │
              │  - Inference Endpoint │
              │  - Model Registry     │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   AWS S3              │
              │  - Model Artifacts    │
              │  - Training Data      │
              │  - Feedback Data      │
              └───────────────────────┘
```

### Infrastructure Components

- **Application Load Balancer (ALB)**: Routes traffic to API Gateway
- **ECS Fargate**: Hosts all microservices containers
- **SageMaker**: Training, inference endpoints, model registry
- **S3**: Model artifacts, training data, feedback storage
- **DynamoDB**: Terraform state locking
- **CloudWatch**: Logging and monitoring
- **ECR**: Container image registry
- **Service Discovery**: Internal service communication

---

## Service Responsibilities

### 1. API Gateway Service (`api-gateway-service`)
**Port:** 8080  
**Responsibility:** Unified entry point for all client requests
- Routes requests to appropriate microservices
- Handles CORS for web clients
- Request logging and correlation IDs
- Health check aggregation

### 2. Inference Service (`inference-service`)
**Port:** 8000  
**Responsibility:** Real-time sentiment predictions
- Preprocesses text using tokenizer from S3
- Invokes SageMaker inference endpoint
- Returns sentiment label (positive/negative/neutral) with confidence scores
- Handles tokenizer caching and fallback mechanisms

### 3. Feedback Service (`feedback-service`)
**Port:** 8001  
**Responsibility:** Collects and stores user feedback
- Accepts feedback on model predictions
- Stores feedback locally (JSON) and in S3
- Tracks prediction vs. actual label discrepancies
- Provides feedback data for retraining

### 4. Model Registry Service (`model-registry-service`)
**Port:** 8002  
**Responsibility:** Model versioning and tracking
- Queries SageMaker for active models and endpoints
- Tracks model versions, accuracy metrics, and deployment status
- Provides model listing and active model identification
- Integrates with SageMaker Model Registry

### 5. Evaluation Service (`evaluation-service`)
**Port:** 8003  
**Responsibility:** Model performance evaluation
- Loads feedback data from S3
- Calculates accuracy metrics on feedback dataset
- Generates evaluation reports
- Tracks model performance over time

### 6. Retraining Service (`retraining-service`)
**Port:** 8004  
**Responsibility:** Model retraining orchestration
- Loads feedback data from S3
- Prepares training dataset (combines original + feedback)
- Triggers SageMaker training jobs
- Calculates validation accuracy and confusion matrix
- Returns training metrics

### 7. Notification Service (`notification-service`)
**Port:** 8005  
**Responsibility:** Event notifications
- Publishes notifications to SNS
- Handles training completion, deployment, and error events
- Supports email/SMS notifications (via SNS)

### 8. Model Init Service (`model-init-service`)
**Port:** 8006  
**Responsibility:** Initial model bootstrap and deployment
- Triggers initial SageMaker training jobs
- Monitors training job status
- Deploys trained models to SageMaker endpoints
- **Auto-deployment**: Automatically deploys model when training completes (optional)
- Manages endpoint lifecycle (create/update)

---

## API Specifications

### Base URL
- **Production:** `http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com`
- **Local:** `http://localhost:8080`
- **Swagger UI:** `{BASE_URL}/docs`

### Endpoints

#### 1. Health Check
```
GET /health
```
**Response:**
```json
{
  "status": "ok",
  "services": {
    "inference": "ok",
    "feedback": "ok",
    "model-registry": "ok"
  }
}
```

#### 2. Predict Sentiment
```
POST /predict
```
**Request:**
```json
{
  "text": "This product is amazing!"
}
```
**Response:**
```json
{
  "label": "positive",
  "confidence": 0.95
}
```
**Error Handling:**
- `503 Service Unavailable`: SageMaker endpoint not ready
- `500 Internal Server Error`: Tokenizer loading failure

#### 3. Submit Feedback
```
POST /feedback
```
**Request:**
```json
{
  "text": "This product is amazing!",
  "model_prediction": "positive",
  "user_label": "positive"
}
```
**Response:**
```json
{
  "message": "Feedback submitted",
  "id": 0
}
```

#### 4. List Models
```
GET /models
```
**Response:**
```json
{
  "models": [
    {
      "version": "ml-sentiment-model-20251214-120000",
      "path": "s3://ml-sentiment-models-.../training-output/.../model.tar.gz",
      "accuracy": null,
      "is_active": true,
      "source": "sagemaker",
      "endpoint_status": "InService"
    }
  ],
  "count": 1
}
```

#### 5. Run Evaluation
```
POST /evaluate
```
**Response:**
```json
{
  "accuracy": 0.85,
  "total_samples": 100,
  "correct_predictions": 85,
  "feedback_count": 100
}
```

#### 6. Trigger Retraining
```
POST /retrain
```
**Response:**
```json
{
  "message": "Retraining triggered",
  "training_job_name": "ml-sentiment-training-20251214-120000",
  "test_accuracy": 0.82,
  "validation_accuracy": 0.85,
  "confusion_matrix": [[45, 5, 2], [3, 38, 4], [1, 2, 0]],
  "confusion_matrix_labels": ["positive", "negative", "neutral"]
}
```
**Error Handling:**
- `400 Bad Request`: Insufficient feedback data (< 10 samples)

#### 7. Bootstrap Training (Initial Model)
```
POST /model-init/bootstrap?auto_deploy=true
```
**Query Parameters:**
- `auto_deploy` (optional, default: `false`): Automatically deploy model when training completes

**Response:**
```json
{
  "message": "SageMaker training job started",
  "training_job_name": "ml-sentiment-training-20251214-120000",
  "status": "InProgress",
  "estimated_time": "15-20 minutes",
  "auto_deploy": true,
  "note": "Auto-deployment enabled. Model will be deployed automatically when training completes."
}
```

#### 8. Check Training Status
```
GET /model-init/status/{job_name}
```
**Response:**
```json
{
  "job_name": "ml-sentiment-training-20251214-120000",
  "status": "Completed",
  "creation_time": "2025-12-14T12:00:00Z",
  "training_end_time": "2025-12-14T12:18:00Z",
  "model_artifacts": "s3://ml-sentiment-models-.../model.tar.gz"
}
```

#### 9. Deploy Model
```
POST /model-init/deploy/{job_name}
```
**Response:**
```json
{
  "message": "Model deployment updated",
  "endpoint_name": "ml-sentiment-endpoint",
  "model_name": "ml-sentiment-model-20251214-120000",
  "status": "Creating",
  "estimated_time": "3-5 minutes"
}
```

#### 10. Check Endpoint Status
```
GET /model-init/endpoint-status
```
**Response:**
```json
{
  "endpoint_name": "ml-sentiment-endpoint",
  "status": "InService",
  "creation_time": "2025-12-14T12:00:00Z",
  "last_modified_time": "2025-12-14T12:20:00Z"
}
```

### Error Response Format
```json
{
  "error": "Error message",
  "detail": "Additional error details"
}
```

---

## CI/CD Workflow

### GitHub Actions Pipeline (`.github/workflows/ci.yml`)

The CI/CD pipeline runs on:
- Push to `main` branch
- Pull requests to `main` branch
- Manual workflow dispatch

#### Pipeline Stages

**1. Validate and Build**
- Validates project structure (all services exist)
- Builds Docker images for all microservices
- Pushes images to AWS ECR
- Tags images with Git commit SHA

**2. Deploy Infrastructure**
- Configures AWS credentials via OIDC
- Uploads training data and SageMaker scripts to S3
- Runs Terraform plan and apply
- Handles Terraform state locking (DynamoDB)
- Retries with exponential backoff on lock conflicts
- Updates ECS services with new task definitions

**3. Post-Deployment**
- Waits for services to become healthy
- Validates service health endpoints
- Reports deployment status

### CI Workflow Features

- **Tests on Push/PR**: Validates service structure and builds images
- **Docker Image Building**: Multi-stage builds for each microservice
- **Infrastructure as Code**: Terraform manages all AWS resources
- **State Locking**: DynamoDB prevents concurrent Terraform runs
- **Automatic Deployment**: Changes to `main` branch automatically deploy to AWS
- **Rollback Capability**: Previous task definitions remain available

### Workflow Diagram

```
Push to main
    │
    ▼
┌─────────────────┐
│  Validate Code  │
│  Build Images   │
│  Push to ECR    │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Terraform Plan │
│  (Infrastructure)│
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│ Terraform Apply │
│  (Deploy AWS)   │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│ Update ECS      │
│ Services        │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│ Health Checks   │
│ Validation      │
└─────────────────┘
```

---

## System Overview

### Technology Stack

**Backend:**
- Python 3.11
- FastAPI (microservices framework)
- PyTorch + HuggingFace Transformers (ML framework)
- Pre-trained model: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- boto3 (AWS SDK)

**Infrastructure:**
- AWS ECS Fargate (container orchestration)
- AWS SageMaker (ML training and inference)
- AWS S3 (object storage)
- AWS ALB (load balancing)
- AWS ECR (container registry)
- Terraform (Infrastructure as Code)

**CI/CD:**
- GitHub Actions
- Docker
- Terraform

### Data Flow

1. **Inference Flow:**
   ```
   Client → API Gateway → Inference Service → SageMaker Endpoint → Response
   ```

2. **Feedback Flow:**
   ```
   Client → API Gateway → Feedback Service → S3 Storage
   ```

3. **Retraining Flow:**
   ```
   Retraining Service → Load Feedback from S3 → Trigger SageMaker Training → 
   Model Artifacts to S3 → Deploy to Endpoint
   ```

4. **Auto-Deployment Flow:**
   ```
   Bootstrap Training → Background Monitor → Training Complete → 
   Auto-Deploy to Endpoint
   ```

---

## Setup Instructions

### Prerequisites

- AWS Account with appropriate permissions
- AWS CLI configured (`aws configure`)
- Terraform >= 1.5.0
- Docker and Docker Compose
- Python 3.11+
- Git

### Local Development Setup

1. **Clone Repository:**
   ```bash
   git clone <repository-url>
   cd ml-sentiment-feedback-loop
   ```

2. **Set Environment Variables:**
   ```bash
   export AWS_REGION=us-east-2
   export AWS_ACCOUNT_ID=your-account-id
   ```

3. **Build and Run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

4. **Access Services:**
   - API Gateway: http://localhost:8080
   - Swagger UI: http://localhost:8080/docs
   - Individual services: localhost:8000-8006

### AWS Deployment Setup

1. **Configure AWS Credentials:**
   ```bash
   aws configure
   # Or use IAM roles for EC2/ECS
   ```

2. **Initialize Terraform:**
   ```bash
   cd infrastructure
   terraform init
   ```

3. **Upload Training Data:**
   ```bash
   aws s3 cp Dataset/train_data.csv s3://ml-sentiment-data-{ACCOUNT_ID}/train_data.csv
   ```

4. **Upload SageMaker Scripts:**
   ```bash
   aws s3 sync sagemaker-scripts/ s3://ml-sentiment-models-{ACCOUNT_ID}/sagemaker-scripts/
   ```

5. **Deploy Infrastructure:**
   ```bash
   terraform plan
   terraform apply
   ```

6. **Get ALB URL:**
   ```bash
   terraform output alb_url
   ```

### GitHub Actions Setup

1. **Configure AWS OIDC:**
   - Create IAM role with OIDC provider for GitHub
   - Update `AWS_ROLE_ARN` in `.github/workflows/ci.yml`

2. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

3. **Monitor Deployment:**
   - Check GitHub Actions tab for pipeline status
   - View CloudWatch logs for service health

---

## Default Accounts and Test Data

### Test Data

**Training Data Location:**
- S3: `s3://ml-sentiment-data-{ACCOUNT_ID}/train_data.csv`
- Format: CSV with columns: `text`, `label` (Positive/Negative/Neutral)

**Sample Training Data:**
```csv
text,label
"This product is amazing!",Positive
"Terrible quality, very disappointed",Negative
"It's okay, nothing special",Neutral
```

### Test Endpoints

**1. Test Prediction:**
```bash
curl -X POST http://{ALB_URL}/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

**2. Submit Feedback:**
```bash
curl -X POST http://{ALB_URL}/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is amazing!",
    "model_prediction": "positive",
    "user_label": "positive"
  }'
```

**3. Trigger Retraining (after 10+ feedback samples):**
```bash
curl -X POST http://{ALB_URL}/retrain
```

**4. Bootstrap with Auto-Deploy:**
```bash
curl -X POST "http://{ALB_URL}/model-init/bootstrap?auto_deploy=true"
```

### Default Configuration

- **Region:** `us-east-2`
- **SageMaker Instance:** `ml.m5.large` (training), `ml.t2.medium` (inference)
- **ECS Task Count:** 1 per service
- **ALB Health Check:** `/health` endpoint
- **Service Discovery:** `ml-sentiment.local` domain

---

## Repository Links

### GitHub Repository
```
https://github.com/{your-username}/ml-sentiment-feedback-loop
```

### Key Directories

- `/services` - Microservices source code
- `/infrastructure` - Terraform IaC
- `/sagemaker-scripts` - SageMaker training/inference scripts
- `/.github/workflows` - CI/CD pipeline definitions
- `/aws-slop` - Deployment scripts and documentation

### Documentation

- **Swagger UI:** `{ALB_URL}/docs`
- **SageMaker Guide:** `aws-slop/docs/SAGEMAKER_GUIDE.md`

---

## Quick Start

### 1. Deploy Infrastructure
```bash
cd infrastructure
terraform apply
```

### 2. Upload Training Data
```bash
aws s3 cp Dataset/train_data.csv s3://ml-sentiment-data-{ACCOUNT_ID}/train_data.csv
```

### 3. Bootstrap Initial Model (with auto-deploy)
```bash
curl -X POST "http://{ALB_URL}/model-init/bootstrap?auto_deploy=true"
```

### 4. Wait for Training (15-20 minutes)
- Model will automatically deploy when training completes

### 5. Test Prediction
```bash
curl -X POST http://{ALB_URL}/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

---

## Troubleshooting

### Common Issues

**1. Training Job Fails:**
- Check CloudWatch logs for training job
- Verify `train_data.csv` exists in S3
- Check SageMaker execution role permissions

**2. Endpoint Not Ready:**
- Wait 3-5 minutes after deployment
- Check endpoint status: `GET /model-init/endpoint-status`
- View CloudWatch logs for endpoint creation

**3. Services Return 503:**
- Services may still be starting (wait 2-3 minutes)
- Check ECS service status in AWS Console
- Verify ALB target group health

**4. Auto-Deployment Not Working:**
- Ensure `auto_deploy=true` parameter is set
- Check background task logs in CloudWatch
- Verify training job completed successfully

---

## License

MIT License
