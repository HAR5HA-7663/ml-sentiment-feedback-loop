from fastapi import FastAPI, Request
from pydantic import BaseModel
import json
import os
import boto3
from datetime import datetime
from pathlib import Path
from app.logging_middleware import RequestLoggingMiddleware, log_info

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)

FEEDBACK_FILE = Path("/app/feedback.json")
S3_BUCKET = os.getenv("S3_DATA_BUCKET", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

# Initialize S3 client
s3_client = boto3.client('s3', region_name=AWS_REGION) if S3_BUCKET else None


class FeedbackRequest(BaseModel):
    text: str
    model_prediction: str
    user_label: str


@app.get("/health")
async def health():
    return {"status": "ok", "s3_enabled": bool(S3_BUCKET)}


def load_feedback():
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return []


def save_feedback(feedback_list):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_list, f, indent=2)


def save_feedback_to_s3(entry, feedback_id):
    """Save feedback entry to S3"""
    if not s3_client or not S3_BUCKET:
        return False
    
    try:
        # Save individual feedback entry
        timestamp = datetime.utcnow().isoformat()
        key = f"feedback/{timestamp}_{feedback_id}.json"
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(entry),
            ContentType='application/json'
        )
        
        log_info("feedback-service", f"Saved feedback to S3: {key}")
        return True
    except Exception as e:
        log_info("feedback-service", f"Error saving to S3: {e}")
        return False


@app.post("/submit-feedback")
async def submit_feedback(request: Request, req_body: FeedbackRequest):
    request_id = getattr(request.state, "request_id", "unknown")
    
    feedback_list = load_feedback()
    
    entry = {
        "text": req_body.text,
        "model_prediction": req_body.model_prediction,
        "user_label": req_body.user_label,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    feedback_list.append(entry)
    save_feedback(feedback_list)
    
    feedback_id = len(feedback_list) - 1
    
    # Also save to S3
    save_feedback_to_s3(entry, feedback_id)
    
    log_info(request_id, f"Feedback stored (ID: {feedback_id}) | Prediction: {req_body.model_prediction}, True Label: {req_body.user_label}")
    
    return {"message": "Feedback submitted", "id": feedback_id}


