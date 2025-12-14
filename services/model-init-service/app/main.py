from fastapi import FastAPI, Request
import os
import boto3
import time
from datetime import datetime
from app.logging_middleware import RequestLoggingMiddleware, log_info

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
MODELS_BUCKET = os.getenv("S3_MODELS_BUCKET", "")
DATA_BUCKET = os.getenv("S3_DATA_BUCKET", "")
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "")
PROJECT_NAME = os.getenv("PROJECT_NAME", "ml-sentiment")

# Initialize clients
sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sagemaker_enabled": bool(SAGEMAKER_ROLE_ARN),
        "data_bucket": DATA_BUCKET,
        "models_bucket": MODELS_BUCKET
    }


@app.post("/bootstrap")
async def bootstrap(request: Request):
    """
    Bootstrap initial model by triggering SageMaker training job
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    try:
        log_info(request_id, "Starting SageMaker model bootstrap")
        
        # Check if training data exists in S3
        data_key = "train_data.csv"
        log_info(request_id, f"Checking for training data: s3://{DATA_BUCKET}/{data_key}")
        
        try:
            s3_client.head_object(Bucket=DATA_BUCKET, Key=data_key)
            log_info(request_id, "Training data found in S3")
        except:
            return {
                "error": "Training data not found in S3",
                "message": f"Please upload train_data.csv to s3://{DATA_BUCKET}/{data_key}"
            }
        
        # Create unique job name
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        training_job_name = f"{PROJECT_NAME}-training-{timestamp}"
        
        log_info(request_id, f"Creating SageMaker training job: {training_job_name}")
        
        # Training job configuration
        training_config = {
            'TrainingJobName': training_job_name,
            'RoleArn': SAGEMAKER_ROLE_ARN,
            'AlgorithmSpecification': {
                'TrainingImage': f'763104351884.dkr.ecr.{AWS_REGION}.amazonaws.com/tensorflow-training:2.11-cpu-py39',
                'TrainingInputMode': 'File',
                'EnableSageMakerMetricsTimeSeries': True
            },
            'InputDataConfig': [
                {
                    'ChannelName': 'train',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': f's3://{DATA_BUCKET}/',
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/csv'
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': f's3://{MODELS_BUCKET}/training-output/'
            },
            'ResourceConfig': {
                'InstanceType': 'ml.m5.large',
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 3600
            },
            'HyperParameters': {
                'epochs': '5',
                'batch-size': '32',
                'sagemaker_program': 'train.py',
                'sagemaker_submit_directory': f's3://{MODELS_BUCKET}/sagemaker-scripts/sourcedir.tar.gz',
                'sagemaker_region': AWS_REGION
            },
            'Tags': [
                {'Key': 'Project', 'Value': PROJECT_NAME},
                {'Key': 'Type', 'Value': 'InitialModel'}
            ]
        }
        
        # Start training job
        response = sagemaker_client.create_training_job(**training_config)
        
        log_info(request_id, f"Training job created: {training_job_name}")
        log_info(request_id, "Training will take approximately 15-20 minutes")
        
        return {
            "message": "SageMaker training job started",
            "training_job_name": training_job_name,
            "status": "InProgress",
            "estimated_time": "15-20 minutes",
            "note": "Use /status endpoint to check progress"
        }
    
    except Exception as e:
        log_info(request_id, f"Error: {str(e)}")
        return {"error": str(e)}


@app.get("/status/{job_name}")
async def get_training_status(job_name: str):
    """Check status of a training job"""
    try:
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        
        status = response['TrainingJobStatus']
        
        result = {
            "job_name": job_name,
            "status": status,
            "creation_time": str(response.get('CreationTime', '')),
            "training_start_time": str(response.get('TrainingStartTime', '')),
            "training_end_time": str(response.get('TrainingEndTime', ''))
        }
        
        if status == 'Completed':
            result['model_artifacts'] = response.get('ModelArtifacts', {}).get('S3ModelArtifacts', '')
            result['message'] = 'Training completed! Model is ready.'
        elif status == 'Failed':
            result['failure_reason'] = response.get('FailureReason', 'Unknown')
        elif status == 'InProgress':
            result['message'] = 'Training in progress...'
        
        return result
    
    except Exception as e:
        return {"error": str(e)}


@app.post("/deploy/{job_name}")
async def deploy_model(job_name: str, request: Request):
    """Deploy trained model to SageMaker endpoint"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    try:
        log_info(request_id, f"Deploying model from job: {job_name}")
        
        # Get training job details
        training_job = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        
        if training_job['TrainingJobStatus'] != 'Completed':
            return {"error": "Training job not completed yet"}
        
        model_data = training_job['ModelArtifacts']['S3ModelArtifacts']
        
        # Create model
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        model_name = f"{PROJECT_NAME}-model-{timestamp}"
        
        log_info(request_id, f"Creating SageMaker model: {model_name}")
        
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': f'763104351884.dkr.ecr.{AWS_REGION}.amazonaws.com/tensorflow-inference:2.11-cpu',
                'ModelDataUrl': model_data,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': f's3://{MODELS_BUCKET}/sagemaker-scripts/sourcedir.tar.gz',
                    'SAGEMAKER_REGION': AWS_REGION
                }
            },
            ExecutionRoleArn=SAGEMAKER_ROLE_ARN
        )
        
        # Create endpoint configuration
        endpoint_config_name = f"{PROJECT_NAME}-endpoint-config-{timestamp}"
        
        log_info(request_id, f"Creating endpoint configuration: {endpoint_config_name}")
        
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InstanceType': 'ml.t2.medium',
                    'InitialInstanceCount': 1,
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        
        # Create or update endpoint
        endpoint_name = f"{PROJECT_NAME}-endpoint"
        
        try:
            # Try to update existing endpoint
            log_info(request_id, f"Updating endpoint: {endpoint_name}")
            sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            action = "updated"
        except sagemaker_client.exceptions.ClientError:
            # Create new endpoint
            log_info(request_id, f"Creating new endpoint: {endpoint_name}")
            sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            action = "created"
        
        log_info(request_id, f"Endpoint {action}: {endpoint_name}")
        
        return {
            "message": f"Model deployment {action}",
            "endpoint_name": endpoint_name,
            "model_name": model_name,
            "status": "Creating",
            "estimated_time": "3-5 minutes",
            "note": "Endpoint will be available shortly"
        }
    
    except Exception as e:
        log_info(request_id, f"Deployment error: {str(e)}")
        return {"error": str(e)}


@app.get("/endpoint-status")
async def get_endpoint_status():
    """Check if endpoint is ready"""
    try:
        endpoint_name = f"{PROJECT_NAME}-endpoint"
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        
        return {
            "endpoint_name": endpoint_name,
            "status": response['EndpointStatus'],
            "creation_time": str(response.get('CreationTime', '')),
            "last_modified_time": str(response.get('LastModifiedTime', ''))
        }
    except sagemaker_client.exceptions.ClientError as e:
        if 'Could not find endpoint' in str(e):
            return {"status": "NotFound", "message": "Endpoint does not exist yet"}
        return {"error": str(e)}


