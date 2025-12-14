from fastapi import FastAPI, Request
from pydantic import BaseModel
import boto3
import json
import os
from app.logging_middleware import RequestLoggingMiddleware, log_info

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)

SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT_NAME", "ml-sentiment-endpoint")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

# Initialize SageMaker Runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    label: str
    confidence: float


@app.get("/health")
async def health():
    return {"status": "ok", "sagemaker_endpoint": SAGEMAKER_ENDPOINT}


async def predict_sentiment(text: str) -> tuple[str, float]:
    """Call SageMaker endpoint for prediction"""
    try:
        # Prepare payload for SageMaker
        payload = {
            "instances": [{"text": text}]
        }
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        result = json.loads(response['Body'].read().decode())
        
        # Parse SageMaker response
        if 'predictions' in result:
            prediction = result['predictions'][0]
            label = prediction.get('label', 'neutral')
            confidence = prediction.get('confidence', 0.5)
        else:
            label = result.get('label', 'neutral')
            confidence = result.get('confidence', 0.5)
        
        return label, confidence
        
    except Exception as e:
        log_info("inference", f"SageMaker error: {e}, falling back to neutral")
        # Fallback to neutral if SageMaker fails
        return "neutral", 0.5


@app.post("/predict-sentiment", response_model=SentimentResponse)
async def predict_sentiment_endpoint(request: Request, req_body: SentimentRequest):
    request_id = getattr(request.state, "request_id", "unknown")
    
    log_info(request_id, f"Predicting sentiment for text (length: {len(req_body.text)} chars)")
    log_info(request_id, f"Using SageMaker endpoint: {SAGEMAKER_ENDPOINT}")
    
    label, confidence = await predict_sentiment(req_body.text)
    
    log_info(request_id, f"Prediction: {label} (confidence: {confidence:.4f})")
    
    return SentimentResponse(label=label, confidence=confidence)


