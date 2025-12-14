from fastapi import FastAPI, Request
from pydantic import BaseModel
import json
import os
import boto3
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from app.logging_middleware import RequestLoggingMiddleware, log_info

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)

MODELS_FILE = Path("/app/models.json")

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
PROJECT_NAME = os.getenv("PROJECT_NAME", "ml-sentiment")
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", f"{PROJECT_NAME}-endpoint")

# Initialize SageMaker client
sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION) if AWS_REGION else None


class ModelRegister(BaseModel):
    version: str
    path: str
    accuracy: float
    is_active: bool


class ModelEntry(BaseModel):
    version: str
    path: str
    accuracy: float
    is_active: bool


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sagemaker_enabled": sagemaker_client is not None,
        "endpoint_name": SAGEMAKER_ENDPOINT_NAME
    }


def load_models():
    if MODELS_FILE.exists():
        with open(MODELS_FILE, "r") as f:
            return json.load(f)
    return []


def save_models(models_list):
    with open(MODELS_FILE, "w") as f:
        json.dump(models_list, f, indent=2)


@app.post("/register")
async def register_model(request: Request, model: ModelRegister):
    request_id = getattr(request.state, "request_id", "unknown")
    
    models_list = load_models()
    
    if model.is_active:
        for m in models_list:
            m["is_active"] = False
        log_info(request_id, f"Deactivating previous active models")
    
    entry = {
        "version": model.version,
        "path": model.path,
        "accuracy": model.accuracy,
        "is_active": model.is_active
    }
    
    models_list.append(entry)
    save_models(models_list)
    
    log_info(request_id, f"Model registered: {model.version} | Accuracy: {model.accuracy:.4f} | Active: {model.is_active}")
    
    return {"message": "Model registered", "version": model.version}


def get_sagemaker_models() -> List[Dict]:
    """Query SageMaker for models, endpoints, and training jobs"""
    sagemaker_models = []
    
    if not sagemaker_client:
        log_info("model-registry", "SageMaker client not initialized (AWS_REGION not set?)")
        return sagemaker_models
    
    log_info("model-registry", f"Querying SageMaker: endpoint={SAGEMAKER_ENDPOINT_NAME}, project={PROJECT_NAME}")
    
    try:
        # Get SageMaker Endpoint info
        try:
            log_info("model-registry", f"Describing endpoint: {SAGEMAKER_ENDPOINT_NAME}")
            endpoint_response = sagemaker_client.describe_endpoint(EndpointName=SAGEMAKER_ENDPOINT_NAME)
            endpoint_config_name = endpoint_response.get('EndpointConfigName')
            log_info("model-registry", f"Endpoint config: {endpoint_config_name}")
            
            if endpoint_config_name:
                config_response = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
                model_name = config_response['ProductionVariants'][0].get('ModelName')
                log_info("model-registry", f"Model name: {model_name}")
                
                if model_name:
                    model_response = sagemaker_client.describe_model(ModelName=model_name)
                    creation_time = model_response.get('CreationTime', datetime.now())
                    
                    # Get model artifacts from model response
                    model_artifacts = model_response.get('PrimaryContainer', {}).get('ModelDataUrl', 'N/A')
                    
                    sagemaker_models.append({
                        "version": f"sagemaker-{model_name}",
                        "path": model_artifacts,
                        "accuracy": None,  # SageMaker doesn't store accuracy in model metadata
                        "is_active": endpoint_response.get('EndpointStatus') == 'InService',
                        "source": "sagemaker",
                        "endpoint_status": endpoint_response.get('EndpointStatus'),
                        "created_at": creation_time.isoformat() if hasattr(creation_time, 'isoformat') else str(creation_time),
                        "model_name": model_name
                    })
                    log_info("model-registry", f"Added SageMaker model: {model_name}")
        except sagemaker_client.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_msg = e.response.get('Error', {}).get('Message', str(e))
            log_info("model-registry", f"Error querying SageMaker endpoint ({error_code}): {error_msg}")
        except Exception as e:
            log_info("model-registry", f"Unexpected error querying endpoint: {e}")
        
        # Get recent training jobs
        try:
            log_info("model-registry", f"Listing training jobs with prefix: {PROJECT_NAME}")
            training_jobs = sagemaker_client.list_training_jobs(
                NameContains=PROJECT_NAME,
                MaxResults=10,
                SortBy='CreationTime',
                SortOrder='Descending'
            )
            
            job_count = len(training_jobs.get('TrainingJobSummaries', []))
            log_info("model-registry", f"Found {job_count} training jobs")
            
            for job in training_jobs.get('TrainingJobSummaries', []):
                if job['TrainingJobStatus'] == 'Completed':
                    job_name = job['TrainingJobName']
                    try:
                        job_details = sagemaker_client.describe_training_job(TrainingJobName=job_name)
                        creation_time = job_details.get('CreationTime', datetime.now())
                        
                        # Extract model artifacts
                        model_artifacts = job_details.get('ModelArtifacts', {}).get('S3ModelArtifacts', '')
                        
                        sagemaker_models.append({
                            "version": f"training-{job_name}",
                            "path": model_artifacts,
                            "accuracy": None,  # Would need to parse from CloudWatch logs
                            "is_active": False,
                            "source": "sagemaker-training",
                            "training_job_name": job_name,
                            "status": job['TrainingJobStatus'],
                            "created_at": creation_time.isoformat() if hasattr(creation_time, 'isoformat') else str(creation_time)
                        })
                        log_info("model-registry", f"Added training job: {job_name}")
                    except Exception as e:
                        log_info("model-registry", f"Error getting training job details for {job_name}: {e}")
        except Exception as e:
            log_info("model-registry", f"Error listing training jobs: {e}")
            
    except Exception as e:
        log_info("model-registry", f"Error querying SageMaker: {e}")
    
    return sagemaker_models


@app.get("/models")
async def get_models(request: Request):
    request_id = getattr(request.state, "request_id", "unknown")
    
    log_info(request_id, f"GET /models - AWS_REGION={AWS_REGION}, ENDPOINT={SAGEMAKER_ENDPOINT_NAME}, PROJECT={PROJECT_NAME}")
    log_info(request_id, f"SageMaker client initialized: {sagemaker_client is not None}")
    
    # Load local registry models
    local_models = load_models()
    log_info(request_id, f"Loaded {len(local_models)} local models")
    
    # Get SageMaker models
    sagemaker_models = get_sagemaker_models()
    log_info(request_id, f"Found {len(sagemaker_models)} SageMaker models")
    
    # Combine both sources
    all_models = local_models + sagemaker_models
    
    log_info(request_id, f"Returning {len(all_models)} models ({len(local_models)} local, {len(sagemaker_models)} SageMaker)")
    
    return {
        "models": all_models,
        "count": len(all_models),
        "local_count": len(local_models),
        "sagemaker_count": len(sagemaker_models)
    }


@app.get("/active-model")
async def get_active_model():
    models_list = load_models()
    active = next((m for m in models_list if m.get("is_active")), None)
    
    if active is None:
        return {"error": "No active model found"}
    
    return active

