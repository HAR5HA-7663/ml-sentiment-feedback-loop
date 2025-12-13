from fastapi import FastAPI, Request
from pydantic import BaseModel
import json
from pathlib import Path
from datetime import datetime
from app.logging_middleware import RequestLoggingMiddleware, log_info

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)

NOTIFICATIONS_FILE = Path("/app/notifications.json")


class NotificationRequest(BaseModel):
    event: str
    details: dict


@app.get("/health")
async def health():
    return {"status": "ok"}


def load_notifications():
    if NOTIFICATIONS_FILE.exists():
        with open(NOTIFICATIONS_FILE, "r") as f:
            return json.load(f)
    return []


def save_notifications(notifications_list):
    with open(NOTIFICATIONS_FILE, "w") as f:
        json.dump(notifications_list, f, indent=2)


@app.post("/notify")
async def notify(request: Request, req_body: NotificationRequest):
    request_id = getattr(request.state, "request_id", "unknown")
    
    notifications_list = load_notifications()
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": req_body.event,
        "details": req_body.details
    }
    
    notifications_list.append(entry)
    save_notifications(notifications_list)
    
    log_info(request_id, f"Notification recorded: {req_body.event}")
    
    return {"message": "Notification recorded", "id": len(notifications_list) - 1}

