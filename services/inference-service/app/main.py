from fastapi import FastAPI, Request
from pydantic import BaseModel
import tensorflow as tf
import pickle
import httpx
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from app.logging_middleware import RequestLoggingMiddleware, log_info

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)

MODELS_DIR = Path("/models")
TOKENIZER_DIR = Path("/models/tokenizers")
REGISTRY_URL = "http://model-registry-service:8002/active-model"

MAX_SEQUENCE_LENGTH = 200
model = None
tokenizer = None
current_model_version = None

label_map = {0: "Positive", 1: "Negative", 2: "Neutral"}


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    label: str
    confidence: float


async def load_model_from_registry():
    global model, tokenizer, current_model_version
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(REGISTRY_URL, timeout=5.0)
            if response.status_code != 200:
                return False
            
            active_model = response.json()
            if not active_model or "error" in active_model:
                return False
            
            model_path_str = active_model.get("path", "")
            if not model_path_str:
                return False
            
            model_path = Path(model_path_str)
            if not model_path.exists():
                return False
            
            version = active_model.get("version", "")
            if version == current_model_version:
                return True
            
            model = tf.keras.models.load_model(str(model_path))
            
            tokenizer_path = TOKENIZER_DIR / f"tokenizer_{version}.pkl"
            if tokenizer_path.exists():
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
            
            current_model_version = version
            print(f"[inference-service] Model loaded: {version}")
            return True
    
    except Exception as e:
        print(f"[inference-service] Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup():
    await load_model_from_registry()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


async def predict_sentiment(text: str) -> tuple[str, float]:
    if model is None or tokenizer is None:
        await load_model_from_registry()
    
    if model is None or tokenizer is None:
        return "neutral", 0.5
    
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    
    prediction = model.predict(padded, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    confidence = float(prediction[predicted_class])
    
    label = label_map.get(predicted_class, "Neutral")
    return label, confidence


@app.post("/predict-sentiment", response_model=SentimentResponse)
async def predict_sentiment_endpoint(request: Request, req_body: SentimentRequest):
    request_id = getattr(request.state, "request_id", "unknown")
    
    log_info(request_id, f"Predicting sentiment for text (length: {len(req_body.text)} chars)")
    log_info(request_id, f"Using model version: {current_model_version or 'unknown'}")
    
    label, confidence = await predict_sentiment(req_body.text)
    
    log_info(request_id, f"Prediction: {label} (confidence: {confidence:.4f})")
    
    return SentimentResponse(label=label, confidence=confidence)

