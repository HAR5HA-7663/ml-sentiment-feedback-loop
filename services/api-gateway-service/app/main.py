from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import requests
import uuid
from typing import Optional
from app.logging_middleware import RequestLoggingMiddleware, log_info

app = FastAPI(
    title="ML Sentiment API Gateway",
    description="Unified entry point for ML Sentiment microservices",
    version="1.0.0"
)
app.add_middleware(RequestLoggingMiddleware)

SERVICE_URLS = {
    "inference": "http://inference-service.ml-sentiment.local:8000",
    "feedback": "http://feedback-service.ml-sentiment.local:8001",
    "model-registry": "http://model-registry-service.ml-sentiment.local:8002",
    "evaluation": "http://evaluation-service.ml-sentiment.local:8003",
    "retraining": "http://retraining-service.ml-sentiment.local:8004",
    "notification": "http://notification-service.ml-sentiment.local:8005",
    "model-init": "http://model-init-service.ml-sentiment.local:8006"
}


class PredictRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    text: str
    model_prediction: str
    user_label: str


def get_or_generate_request_id(request: Request) -> str:
    if hasattr(request.state, "request_id"):
        return request.state.request_id
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id


def forward_request(service_url: str, path: str, method: str, request_id: str, json_data: Optional[dict] = None):
    headers = {"X-Request-ID": request_id}
    url = f"{service_url}{path}"
    
    log_info(request_id, f"Routing {method} request to {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=json_data, headers=headers, timeout=10)
        else:
            raise HTTPException(status_code=405, detail="Method not allowed")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.Timeout:
        print(f"[{request_id}] Timeout calling {url}")
        raise HTTPException(status_code=504, detail=f"Service timeout: {url}")
    
    except requests.exceptions.ConnectionError:
        print(f"[{request_id}] Connection error to {url}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {url}")
    
    except requests.exceptions.HTTPError as e:
        print(f"[{request_id}] HTTP error from {url}: {e.response.status_code}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    
    except Exception as e:
        print(f"[{request_id}] Error calling {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health(request: Request):
    request_id = get_or_generate_request_id(request)
    print(f"[{request_id}] Health check requested")
    
    health_status = {"gateway": "ok", "services": {}}
    
    for service_name, service_url in SERVICE_URLS.items():
        try:
            response = requests.get(f"{service_url}/health", timeout=5)
            health_status["services"][service_name] = {
                "status": "ok" if response.status_code == 200 else "degraded",
                "code": response.status_code
            }
        except Exception as e:
            health_status["services"][service_name] = {
                "status": "down",
                "error": str(e)
            }
    
    all_healthy = all(s.get("status") == "ok" for s in health_status["services"].values())
    health_status["overall"] = "healthy" if all_healthy else "degraded"
    
    return health_status


@app.post("/predict")
async def predict(request: Request, payload: PredictRequest):
    request_id = get_or_generate_request_id(request)
    
    result = forward_request(
        SERVICE_URLS["inference"],
        "/predict-sentiment",
        "POST",
        request_id,
        payload.dict()
    )
    
    log_info(request_id, f"Prediction result: {result.get('label')} (confidence: {result.get('confidence', 0):.2f})")
    
    return result


@app.post("/feedback")
async def submit_feedback(request: Request, payload: FeedbackRequest):
    request_id = get_or_generate_request_id(request)
    
    result = forward_request(
        SERVICE_URLS["feedback"],
        "/submit-feedback",
        "POST",
        request_id,
        payload.dict()
    )
    
    return result


@app.post("/evaluate")
async def evaluate(request: Request):
    request_id = get_or_generate_request_id(request)
    
    result = forward_request(
        SERVICE_URLS["evaluation"],
        "/run-evaluation",
        "POST",
        request_id
    )
    
    return result


@app.post("/retrain")
async def retrain(request: Request):
    request_id = get_or_generate_request_id(request)
    
    result = forward_request(
        SERVICE_URLS["retraining"],
        "/retrain",
        "POST",
        request_id
    )
    
    return result


@app.get("/models")
async def get_models(request: Request):
    request_id = get_or_generate_request_id(request)
    
    result = forward_request(
        SERVICE_URLS["model-registry"],
        "/models",
        "GET",
        request_id
    )
    
    return result


# Model Initialization Service Routes
@app.post("/model-init/bootstrap")
async def bootstrap_model(request: Request):
    request_id = get_or_generate_request_id(request)
    
    result = forward_request(
        SERVICE_URLS["model-init"],
        "/bootstrap",
        "POST",
        request_id
    )
    
    return result


@app.get("/model-init/status/{job_name}")
async def get_training_status(request: Request, job_name: str):
    request_id = get_or_generate_request_id(request)
    
    result = forward_request(
        SERVICE_URLS["model-init"],
        f"/status/{job_name}",
        "GET",
        request_id
    )
    
    return result


@app.post("/model-init/deploy/{job_name}")
async def deploy_model(request: Request, job_name: str):
    request_id = get_or_generate_request_id(request)
    
    result = forward_request(
        SERVICE_URLS["model-init"],
        f"/deploy/{job_name}",
        "POST",
        request_id
    )
    
    return result


@app.get("/model-init/endpoint-status")
async def get_endpoint_status(request: Request):
    request_id = get_or_generate_request_id(request)
    
    result = forward_request(
        SERVICE_URLS["model-init"],
        "/endpoint-status",
        "GET",
        request_id
    )
    
    return result


@app.get("/")
async def root():
    return {
        "service": "ML Sentiment API Gateway",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Make sentiment prediction",
            "POST /feedback": "Submit user feedback",
            "POST /evaluate": "Run model evaluation",
            "POST /retrain": "Trigger model retraining",
            "GET /models": "List all models",
            "GET /health": "Check system health",
            "POST /model-init/bootstrap": "Start SageMaker training",
            "GET /model-init/status/{job_name}": "Check training status",
            "POST /model-init/deploy/{job_name}": "Deploy trained model",
            "GET /model-init/endpoint-status": "Check endpoint status"
        }
    }
