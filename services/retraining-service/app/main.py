from fastapi import FastAPI, Request
import json
import httpx
from datetime import datetime
import os
import boto3
from app.logging_middleware import RequestLoggingMiddleware, log_info

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)

# Service URLs
REGISTRY_URL = "http://model-registry-service.ml-sentiment.local:8002"
NOTIFICATION_URL = "http://notification-service.ml-sentiment.local:8005"

# AWS Configuration
S3_DATA_BUCKET = os.getenv("S3_DATA_BUCKET", "")
S3_MODELS_BUCKET = os.getenv("S3_MODELS_BUCKET", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "")
PROJECT_NAME = os.getenv("PROJECT_NAME", "ml-sentiment")

# SageMaker training container (HuggingFace PyTorch)
TRAINING_IMAGE = f"763104351884.dkr.ecr.{AWS_REGION}.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"

# Initialize AWS clients
s3_client = boto3.client('s3', region_name=AWS_REGION)
sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "s3_data_bucket": S3_DATA_BUCKET,
        "s3_models_bucket": S3_MODELS_BUCKET,
        "sagemaker_role": bool(SAGEMAKER_ROLE_ARN)
    }


def load_feedback_from_s3():
    """Load all feedback from S3"""
    if not S3_DATA_BUCKET:
        log_info("retraining", "S3_DATA_BUCKET not configured")
        return []

    try:
        log_info("retraining", f"Loading feedback from S3 bucket: {S3_DATA_BUCKET}, prefix: feedback/")
        response = s3_client.list_objects_v2(
            Bucket=S3_DATA_BUCKET,
            Prefix='feedback/'
        )

        if 'Contents' not in response:
            log_info("retraining", "No feedback files found in S3")
            return []

        log_info("retraining", f"Found {len(response['Contents'])} feedback files in S3")
        feedback_list = []
        for obj in response['Contents']:
            if obj['Key'].endswith('.json'):
                try:
                    data = s3_client.get_object(Bucket=S3_DATA_BUCKET, Key=obj['Key'])
                    content = json.loads(data['Body'].read().decode('utf-8'))
                    feedback_list.append(content)
                except Exception as e:
                    log_info("retraining", f"Error reading {obj['Key']}: {e}")
                    continue

        log_info("retraining", f"Successfully loaded {len(feedback_list)} feedback entries from S3")
        return feedback_list
    except Exception as e:
        log_info("retraining", f"Error loading feedback from S3: {e}")
        return []


def prepare_training_data(feedback_list):
    """Convert feedback to CSV format for SageMaker training"""
    # Label mapping: Negative=0, Neutral=1, Positive=2
    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

    lines = []
    for item in feedback_list:
        text = item.get('text', '').replace('"', '""')  # Escape quotes
        label = item.get('user_label', 'Neutral')
        label_num = label_map.get(label, 1)
        lines.append(f'"{text}",{label_num}')

    return "text,label\n" + "\n".join(lines)


def upload_training_data(csv_content, job_name):
    """Upload training data to S3"""
    key = f"training/{job_name}/train.csv"
    s3_client.put_object(
        Bucket=S3_DATA_BUCKET,
        Key=key,
        Body=csv_content.encode('utf-8'),
        ContentType='text/csv'
    )
    log_info("retraining", f"Uploaded training data to s3://{S3_DATA_BUCKET}/{key}")
    return f"s3://{S3_DATA_BUCKET}/training/{job_name}"


def create_sagemaker_training_job(job_name, training_data_uri):
    """Create a SageMaker training job"""

    # S3 paths
    output_path = f"s3://{S3_MODELS_BUCKET}/models"
    code_path = f"s3://{S3_MODELS_BUCKET}/sagemaker-scripts/sourcedir.tar.gz"

    training_params = {
        'TrainingJobName': job_name,
        'RoleArn': SAGEMAKER_ROLE_ARN,
        'AlgorithmSpecification': {
            'TrainingImage': TRAINING_IMAGE,
            'TrainingInputMode': 'File',
        },
        'HyperParameters': {
            'epochs': '3',
            'batch_size': '16',
            'learning_rate': '2e-5',
            'sagemaker_program': 'train.py',
            'sagemaker_submit_directory': code_path,
        },
        'InputDataConfig': [
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': training_data_uri,
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'text/csv',
                'InputMode': 'File'
            }
        ],
        'OutputDataConfig': {
            'S3OutputPath': output_path
        },
        'ResourceConfig': {
            'InstanceType': 'ml.m5.large',  # CPU instance (cheaper, sufficient for fine-tuning)
            'InstanceCount': 1,
            'VolumeSizeInGB': 30
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 3600  # 1 hour max
        },
        'Tags': [
            {'Key': 'Project', 'Value': PROJECT_NAME},
            {'Key': 'Service', 'Value': 'retraining'}
        ]
    }

    response = sagemaker_client.create_training_job(**training_params)
    log_info("retraining", f"Created SageMaker training job: {job_name}")
    return response


@app.post("/retrain")
async def retrain(request: Request):
    request_id = getattr(request.state, "request_id", "unknown")

    log_info(request_id, "Starting model retraining via SageMaker")

    # Validate configuration
    if not S3_DATA_BUCKET or not S3_MODELS_BUCKET:
        return {"error": "S3 buckets not configured"}

    if not SAGEMAKER_ROLE_ARN:
        return {"error": "SageMaker role not configured"}

    # Load feedback data
    feedback_list = load_feedback_from_s3()

    if not feedback_list:
        log_info(request_id, "No feedback data available")
        return {"error": "No feedback data available"}

    if len(feedback_list) < 10:
        log_info(request_id, f"Insufficient feedback data: {len(feedback_list)} samples (need 10)")
        return {"error": f"Insufficient feedback data. Have {len(feedback_list)}, need at least 10 samples."}

    log_info(request_id, f"Found {len(feedback_list)} feedback samples")

    # Generate job name
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f"{PROJECT_NAME}-retrain-{timestamp}"

    try:
        # Prepare and upload training data
        csv_content = prepare_training_data(feedback_list)
        training_data_uri = upload_training_data(csv_content, job_name)

        # Create SageMaker training job
        create_sagemaker_training_job(job_name, training_data_uri)

        # Send notification
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                await client.post(
                    f"{NOTIFICATION_URL}/notify",
                    json={
                        "event": "training_started",
                        "details": {
                            "job_name": job_name,
                            "training_samples": len(feedback_list),
                            "timestamp": timestamp
                        }
                    }
                )
            except Exception as e:
                log_info(request_id, f"Failed to send notification: {e}")

        log_info(request_id, f"SageMaker training job created: {job_name}")

        return {
            "message": "Training job started",
            "job_name": job_name,
            "training_samples": len(feedback_list),
            "status": "InProgress",
            "monitor_command": f"aws sagemaker describe-training-job --training-job-name {job_name} --region {AWS_REGION}"
        }

    except Exception as e:
        log_info(request_id, f"Error creating training job: {e}")
        return {"error": str(e)}


@app.get("/training-jobs")
async def list_training_jobs():
    """List recent training jobs"""
    try:
        response = sagemaker_client.list_training_jobs(
            NameContains=PROJECT_NAME,
            MaxResults=10,
            SortBy='CreationTime',
            SortOrder='Descending'
        )

        jobs = []
        for job in response.get('TrainingJobSummaries', []):
            jobs.append({
                'name': job['TrainingJobName'],
                'status': job['TrainingJobStatus'],
                'created': job['CreationTime'].isoformat(),
            })

        return {"jobs": jobs}
    except Exception as e:
        return {"error": str(e)}


@app.get("/training-jobs/{job_name}")
async def get_training_job(job_name: str):
    """Get details of a specific training job"""
    try:
        response = sagemaker_client.describe_training_job(
            TrainingJobName=job_name
        )

        return {
            "name": response['TrainingJobName'],
            "status": response['TrainingJobStatus'],
            "created": response['CreationTime'].isoformat(),
            "secondary_status": response.get('SecondaryStatus'),
            "failure_reason": response.get('FailureReason'),
            "model_artifacts": response.get('ModelArtifacts', {}).get('S3ModelArtifacts'),
            "training_time_seconds": response.get('TrainingTimeInSeconds'),
            "billable_time_seconds": response.get('BillableTimeInSeconds')
        }
    except Exception as e:
        return {"error": str(e)}
