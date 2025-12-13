from fastapi import FastAPI, Request
from pydantic import BaseModel
import json
import os
from pathlib import Path
from app.logging_middleware import RequestLoggingMiddleware, log_info

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)

FEEDBACK_FILE = Path("/app/feedback.json")


class FeedbackRequest(BaseModel):
    text: str
    model_prediction: str
    user_label: str


@app.get("/health")
async def health():
    return {"status": "ok"}


def load_feedback():
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return []


def save_feedback(feedback_list):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_list, f, indent=2)


@app.post("/submit-feedback")
async def submit_feedback(request: Request, req_body: FeedbackRequest):
    request_id = getattr(request.state, "request_id", "unknown")
    
    feedback_list = load_feedback()
    
    entry = {
        "text": req_body.text,
        "model_prediction": req_body.model_prediction,
        "user_label": req_body.user_label
    }
    
    feedback_list.append(entry)
    save_feedback(feedback_list)
    
    feedback_id = len(feedback_list) - 1
    log_info(request_id, f"Feedback stored (ID: {feedback_id}) | Prediction: {req_body.model_prediction}, True Label: {req_body.user_label}")
    
    return {"message": "Feedback submitted", "id": feedback_id}

