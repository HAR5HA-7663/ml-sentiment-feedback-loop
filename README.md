# ML Sentiment Feedback Loop

**Deploy a production-ready MLOps system to your AWS account in minutes using Terraform.**

A complete microservices-based sentiment analysis platform with automated model retraining, evaluation, and deployment - all managed through Infrastructure as Code.

## One-Click AWS Deployment

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ml-sentiment-feedback-loop.git
cd ml-sentiment-feedback-loop

# 2. Configure your AWS credentials
export AWS_PROFILE=your-profile  # or aws configure

# 3. Deploy everything to AWS
cd infrastructure
terraform init
terraform apply

# That's it! Your entire ML platform is now running on AWS.
```

## What Gets Deployed

When you run `terraform apply`, the following infrastructure is automatically created:

| Resource | Description |
|----------|-------------|
| **ECS Fargate Cluster** | 8 containerized microservices |
| **Application Load Balancer** | Public endpoint for API access |
| **SageMaker Endpoint** | Real-time ML inference |
| **S3 Buckets** | Model artifacts + training data |
| **ECR Repositories** | Docker image registry |
| **VPC + Networking** | Subnets, security groups, NAT |
| **Service Discovery** | Internal service communication |
| **CloudWatch** | Logging and monitoring |

**Total deployment time: ~10-15 minutes**

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │      Application Load Balancer      │
                    │         (Public Endpoint)           │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │          API Gateway Service        │
                    │     (Routes to all microservices)   │
                    └─────────────────┬───────────────────┘
                                      │
        ┌─────────────┬───────────────┼───────────────┬─────────────┐
        │             │               │               │             │
        ▼             ▼               ▼               ▼             ▼
   ┌─────────┐  ┌─────────┐    ┌─────────┐    ┌─────────┐   ┌─────────┐
   │Inference│  │Feedback │    │Evaluation│   │Retraining│  │Model    │
   │ Service │  │ Service │    │ Service │    │ Service │   │Registry │
   └────┬────┘  └────┬────┘    └─────────┘    └────┬────┘   └─────────┘
        │            │                             │
        │            ▼                             │
        │       ┌─────────┐                        │
        │       │   S3    │◄───────────────────────┘
        │       │(Feedback│
        │       │  Data)  │
        │       └─────────┘
        │
        ▼
   ┌─────────────────────────────────────┐
   │           AWS SageMaker             │
   │  ┌─────────────┐  ┌──────────────┐  │
   │  │  Training   │  │  Inference   │  │
   │  │    Jobs     │  │   Endpoint   │  │
   │  └─────────────┘  └──────────────┘  │
   └─────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- AWS Account
- [Terraform](https://www.terraform.io/downloads) >= 1.5.0
- [AWS CLI](https://aws.amazon.com/cli/) configured
- [Docker](https://www.docker.com/) (for local development)

### Step 1: Deploy Infrastructure

```bash
cd infrastructure
terraform init
terraform apply
```

Terraform will output your ALB URL:
```
alb_url = "http://ml-sentiment-alb-xxxxxxxxx.us-east-2.elb.amazonaws.com"
```

### Step 2: Bootstrap the ML Model

```bash
# Start training with auto-deployment
curl -X POST "http://<ALB_URL>/model-init/bootstrap?auto_deploy=true"
```

Wait 15-20 minutes for training to complete. The model will automatically deploy to SageMaker.

### Step 3: Make Predictions

```bash
curl -X POST http://<ALB_URL>/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'

# Response: {"label": "Positive", "confidence": 0.95}
```

### Step 4: Collect Feedback & Retrain

```bash
# Submit user feedback
curl -X POST http://<ALB_URL>/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product is amazing!",
    "model_prediction": "Positive",
    "user_label": "Positive"
  }'

# After 10+ feedback samples, trigger retraining
curl -X POST http://<ALB_URL>/retrain
```

## Terraform Infrastructure

All AWS resources are defined in the `infrastructure/` directory:

```
infrastructure/
├── main.tf              # Provider config, VPC, networking
├── ecs.tf               # ECS cluster, services, task definitions
├── alb.tf               # Application Load Balancer, target groups
├── sagemaker.tf         # SageMaker endpoint, IAM roles
├── s3.tf                # S3 buckets for data and models
├── ecr.tf               # Container registries
├── service-discovery.tf # Cloud Map namespace
├── variables.tf         # Configurable parameters
└── outputs.tf           # ALB URL, resource ARNs
```

### Customize Your Deployment

Edit `infrastructure/variables.tf`:

```hcl
variable "aws_region" {
  default = "us-east-2"  # Change to your preferred region
}

variable "project_name" {
  default = "ml-sentiment"  # Change project name
}

variable "ecs_task_memory" {
  default = 1024  # Adjust container memory (MB)
}
```

### Destroy Everything

```bash
cd infrastructure
terraform destroy
```

All resources will be cleanly removed from your AWS account.

## CI/CD Pipeline

Push to `main` branch triggers automatic deployment:

```
git push origin main
    │
    ▼
┌─────────────────────┐
│  Build Docker       │
│  Images (8 services)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Push to ECR        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Terraform Apply    │
│  (Update infra)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Update ECS         │
│  Services           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Health Check       │
│  Validation         │
└─────────────────────┘
```

### Setup GitHub Actions

1. Create IAM OIDC provider for GitHub
2. Create IAM role with required permissions
3. Update `.github/workflows/ci.yml` with your role ARN
4. Push to trigger deployment

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health status of all services |
| `/predict` | POST | Make sentiment prediction |
| `/feedback` | POST | Submit user feedback |
| `/evaluate` | POST | Run model evaluation |
| `/retrain` | POST | Trigger SageMaker retraining |
| `/training-jobs` | GET | List training jobs |
| `/training-jobs/{name}` | GET | Get training job details |
| `/models` | GET | List registered models |
| `/model-init/bootstrap` | POST | Start initial training |
| `/model-init/endpoint-status` | GET | Check SageMaker endpoint |

**Interactive API Docs:** `http://<ALB_URL>/docs`

## Microservices

| Service | Port | Responsibility |
|---------|------|----------------|
| API Gateway | 8080 | Request routing, CORS |
| Inference | 8000 | SageMaker predictions |
| Feedback | 8001 | User feedback collection |
| Model Registry | 8002 | Model versioning |
| Evaluation | 8003 | Performance metrics |
| Retraining | 8004 | SageMaker training jobs |
| Notification | 8005 | SNS notifications |
| Model Init | 8006 | Bootstrap & deployment |

## Technology Stack

- **Infrastructure:** Terraform, AWS (ECS, SageMaker, S3, ALB, ECR)
- **Backend:** Python 3.11, FastAPI
- **ML:** PyTorch, HuggingFace Transformers
- **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **CI/CD:** GitHub Actions, Docker

## Project Structure

```
ml-sentiment-feedback-loop/
├── infrastructure/        # Terraform IaC (deploy everything here)
├── services/              # 8 FastAPI microservices
│   ├── api-gateway-service/
│   ├── inference-service/
│   ├── feedback-service/
│   ├── model-registry-service/
│   ├── evaluation-service/
│   ├── retraining-service/
│   ├── notification-service/
│   └── model-init-service/
├── sagemaker-scripts/     # Training & inference scripts
├── .github/workflows/     # CI/CD pipeline
└── Dataset/               # Sample training data
```

## Troubleshooting

**Services returning 503?**
- Wait 2-3 minutes for ECS services to start
- Check: `curl http://<ALB_URL>/health`

**Training job failed?**
- Check CloudWatch logs for the training job
- Verify S3 bucket permissions

**Endpoint not ready?**
- Check: `curl http://<ALB_URL>/model-init/endpoint-status`
- Wait 3-5 minutes after deployment

## Cost Estimate

Running this infrastructure costs approximately:
- **ECS Fargate:** ~$30-50/month (8 small containers)
- **SageMaker Endpoint:** ~$50-100/month (ml.t2.medium)
- **ALB:** ~$20/month
- **S3/ECR:** ~$5/month

**Tip:** Destroy resources when not in use: `terraform destroy`

## License

MIT License
