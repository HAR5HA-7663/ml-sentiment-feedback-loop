from fastapi import FastAPI, Request, BackgroundTasks
import os
import boto3
import time
import asyncio
import threading
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


def monitor_and_deploy_sync(job_name: str):
    """Background task that monitors training and auto-deploys when complete (sync version for BackgroundTasks)
    
    Uses threading to run async function in background, more reliable than asyncio.run() in FastAPI BackgroundTasks
    """
    def run_async():
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(monitor_and_deploy(job_name))
        except Exception as e:
            log_info(f"auto-deploy-{job_name}", f"Error in background task: {str(e)}")
            import traceback
            log_info(f"auto-deploy-{job_name}", f"Traceback: {traceback.format_exc()}")
        finally:
            loop.close()
    
    # Run in daemon thread so it doesn't block shutdown
    try:
        thread = threading.Thread(target=run_async, daemon=True, name=f"auto-deploy-{job_name}")
        thread.start()
        log_info(f"auto-deploy-{job_name}", f"Background thread started for monitoring {job_name} (thread ID: {thread.ident})")
    except Exception as e:
        log_info(f"auto-deploy-{job_name}", f"Failed to start thread: {str(e)}")
        import traceback
        log_info(f"auto-deploy-{job_name}", f"Traceback: {traceback.format_exc()}")
        raise

async def monitor_and_deploy(job_name: str):
    """Background task that monitors training and auto-deploys when complete"""
    request_id = f"auto-deploy-{job_name}"
    max_wait_time = 3600
    check_interval = 60
    elapsed = 0
    
    log_info(request_id, f"Starting auto-deployment monitor for {job_name}")
    
    while elapsed < max_wait_time:
        try:
            response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']
            
            if status == 'Completed':
                log_info(request_id, f"Training completed. Auto-deploying {job_name}")
                
                model_data = response['ModelArtifacts']['S3ModelArtifacts']
                timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
                model_name = f"{PROJECT_NAME}-model-{timestamp}"
                
                log_info(request_id, f"Creating SageMaker model: {model_name}")
                sagemaker_client.create_model(
                    ModelName=model_name,
                    PrimaryContainer={
                        'Image': f'763104351884.dkr.ecr.{AWS_REGION}.amazonaws.com/tensorflow-training:2.11-cpu-py39',
                        'ModelDataUrl': model_data,
                        'Mode': 'SingleModel',
                        'Environment': {
                            'SAGEMAKER_PROGRAM': 'inference.py',
                            'SAGEMAKER_SUBMIT_DIRECTORY': f's3://{MODELS_BUCKET}/sagemaker-scripts/sourcedir.tar.gz',
                            'SAGEMAKER_REGION': AWS_REGION,
                            'SAGEMAKER_CONTAINER_LOG_LEVEL': '20'
                        }
                    },
                    ExecutionRoleArn=SAGEMAKER_ROLE_ARN
                )
                
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
                
                endpoint_name = f"{PROJECT_NAME}-endpoint"
                try:
                    log_info(request_id, f"Updating endpoint: {endpoint_name}")
                    sagemaker_client.update_endpoint(
                        EndpointName=endpoint_name,
                        EndpointConfigName=endpoint_config_name
                    )
                    action = "updated"
                except sagemaker_client.exceptions.ClientError as e:
                    if 'Could not find endpoint' in str(e) or 'does not exist' in str(e):
                        log_info(request_id, f"Creating new endpoint: {endpoint_name}")
                        sagemaker_client.create_endpoint(
                            EndpointName=endpoint_name,
                            EndpointConfigName=endpoint_config_name
                        )
                        action = "created"
                    else:
                        log_info(request_id, f"Error updating endpoint: {str(e)}")
                        raise
                
                log_info(request_id, f"Auto-deployment complete: endpoint {action}")
                return
                
            elif status == 'Failed':
                log_info(request_id, f"Training failed: {response.get('FailureReason', 'Unknown')}")
                return
                
        except Exception as e:
            log_info(request_id, f"Error checking status: {str(e)}")
            import traceback
            log_info(request_id, f"Traceback: {traceback.format_exc()}")
        
        await asyncio.sleep(check_interval)
        elapsed += check_interval
    
    log_info(request_id, f"Auto-deployment monitor timed out after {max_wait_time}s")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sagemaker_enabled": bool(SAGEMAKER_ROLE_ARN),
        "data_bucket": DATA_BUCKET,
        "models_bucket": MODELS_BUCKET
    }


@app.post("/bootstrap")
async def bootstrap(request: Request, background_tasks: BackgroundTasks, auto_deploy: bool = False):
    """
    Bootstrap initial model by triggering SageMaker training job
    Query params:
    - auto_deploy: If True, automatically deploys model when training completes
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
            'Environment': {
                'MODELS_BUCKET': MODELS_BUCKET,
                'S3_MODELS_BUCKET': MODELS_BUCKET
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
        
        result = {
            "message": "SageMaker training job started",
            "training_job_name": training_job_name,
            "status": "InProgress",
            "estimated_time": "15-20 minutes"
        }
        
        if auto_deploy:
            log_info(request_id, f"Auto-deploy enabled, starting background monitoring thread for {training_job_name}")
            try:
                # Start background monitoring thread immediately (more reliable than BackgroundTasks)
                monitor_and_deploy_sync(training_job_name)
                log_info(request_id, f"Background thread started successfully for {training_job_name}")
                result["auto_deploy"] = True
                result["note"] = "Auto-deployment enabled. Model will be deployed automatically when training completes."
            except Exception as e:
                log_info(request_id, f"Error starting auto-deploy thread: {str(e)}")
                import traceback
                log_info(request_id, f"Traceback: {traceback.format_exc()}")
                result["auto_deploy"] = False
                result["note"] = f"Auto-deployment failed to start: {str(e)}. Please deploy manually."
        else:
            result["note"] = "Use /status endpoint to check progress"
        
        return result
    
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
                'Image': f'763104351884.dkr.ecr.{AWS_REGION}.amazonaws.com/tensorflow-training:2.11-cpu-py39',
                'ModelDataUrl': model_data,
                'Mode': 'SingleModel',
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': f's3://{MODELS_BUCKET}/sagemaker-scripts/sourcedir.tar.gz',
                    'SAGEMAKER_REGION': AWS_REGION,
                    'SAGEMAKER_CONTAINER_LOG_LEVEL': '20'
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


