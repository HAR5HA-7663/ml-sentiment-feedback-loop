from fastapi import FastAPI, Request
from pydantic import BaseModel
import json
from pathlib import Path
from typing import Optional
from app.logging_middleware import RequestLoggingMiddleware, log_info

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)

MODELS_FILE = Path("/app/models.json")


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
    return {"status": "ok"}


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


@app.get("/models")
async def get_models():
    models_list = load_models()
    return {"models": models_list}


@app.get("/active-model")
async def get_active_model():
    models_list = load_models()
    active = next((m for m in models_list if m.get("is_active")), None)
    
    if active is None:
        return {"error": "No active model found"}
    
    return active

