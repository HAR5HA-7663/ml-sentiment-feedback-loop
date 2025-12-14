from fastapi import FastAPI, Request
import json
import os
import boto3
from datetime import datetime
from pathlib import Path
from app.logging_middleware import RequestLoggingMiddleware, log_info

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)

FEEDBACK_FILE = Path("/feedback-data/feedback.json")
EVALUATION_FILE = Path("/app/evaluation.json")
S3_BUCKET = os.getenv("S3_DATA_BUCKET", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

# Initialize S3 client
s3_client = boto3.client('s3', region_name=AWS_REGION) if S3_BUCKET else None


@app.get("/health")
async def health():
    return {"status": "ok", "s3_enabled": bool(S3_BUCKET)}


def load_feedback_from_s3():
    """Load all feedback from S3"""
    if not s3_client or not S3_BUCKET:
        return []
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix='feedback/'
        )
        
        if 'Contents' not in response:
            return []
        
        feedback_list = []
        for obj in response['Contents']:
            try:
                data = s3_client.get_object(Bucket=S3_BUCKET, Key=obj['Key'])
                content = json.loads(data['Body'].read().decode('utf-8'))
                feedback_list.append(content)
            except Exception as e:
                log_info("evaluation", f"Error reading {obj['Key']}: {e}")
                continue
        
        return feedback_list
    except Exception as e:
        log_info("evaluation", f"Error loading feedback from S3: {e}")
        return []


def load_feedback():
    # Try S3 first, fall back to local file
    feedback = load_feedback_from_s3()
    if feedback:
        return feedback
    
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return []


def load_evaluation():
    if EVALUATION_FILE.exists():
        with open(EVALUATION_FILE, "r") as f:
            return json.load(f)
    return None


def save_evaluation(evaluation_data):
    with open(EVALUATION_FILE, "w") as f:
        json.dump(evaluation_data, f, indent=2)


@app.post("/run-evaluation")
async def run_evaluation(request: Request):
    request_id = getattr(request.state, "request_id", "unknown")
    
    log_info(request_id, "Starting model evaluation")
    
    feedback_list = load_feedback()
    
    if not feedback_list:
        log_info(request_id, "No feedback data available for evaluation")
        return {"error": "No feedback data available"}
    
    log_info(request_id, f"Evaluating model on {len(feedback_list)} feedback samples")
    
    correct = sum(1 for entry in feedback_list if entry.get("model_prediction") == entry.get("user_label"))
    total = len(feedback_list)
    accuracy = correct / total if total > 0 else 0.0
    
    evaluation_data = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    save_evaluation(evaluation_data)
    
    log_info(request_id, f"Evaluation complete | Accuracy: {accuracy:.4f} ({correct}/{total} correct)")
    
    return evaluation_data


@app.get("/latest-evaluation")
async def get_latest_evaluation():
    evaluation = load_evaluation()
    
    if evaluation is None:
        return {"error": "No evaluation found"}
    
    return evaluation


