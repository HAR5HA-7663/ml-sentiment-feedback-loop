from fastapi import FastAPI, Request
import json
import httpx
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
from app.logging_middleware import RequestLoggingMiddleware, log_info

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)

FEEDBACK_FILE = Path("/feedback-data/feedback.json")
MODELS_DIR = Path("/models")
TOKENIZER_DIR = Path("/models/tokenizers")
REGISTRY_URL = "http://model-registry-service:8002/register"
NOTIFICATION_URL = "http://notification-service:8005/notify"

MAX_SEQUENCE_LENGTH = 200
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128

label_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
reverse_label_map = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}


@app.get("/health")
async def health():
    return {"status": "ok"}


def load_feedback():
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return []


def load_existing_tokenizer():
    tokenizer_files = sorted(TOKENIZER_DIR.glob("tokenizer_*.pkl"), reverse=True)
    if tokenizer_files:
        with open(tokenizer_files[0], 'rb') as f:
            return pickle.load(f)
    return None


def build_model():
    model = keras.Sequential([
        keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


@app.post("/retrain")
async def retrain(request: Request):
    request_id = getattr(request.state, "request_id", "unknown")
    
    log_info(request_id, "Starting model retraining")
    
    feedback_list = load_feedback()
    
    if not feedback_list:
        log_info(request_id, "No feedback data available")
        return {"error": "No feedback data available"}
    
    if len(feedback_list) < 10:
        log_info(request_id, f"Insufficient feedback data: {len(feedback_list)} samples (need 10)")
        return {"error": "Insufficient feedback data. Need at least 10 samples."}
    
    log_info(request_id, f"Preparing training data from {len(feedback_list)} feedback samples")
    
    texts = [item['text'] for item in feedback_list]
    labels = [item['user_label'] for item in feedback_list]
    
    labels_numeric = [label_map.get(label, 2) for label in labels]
    
    existing_tokenizer = load_existing_tokenizer()
    if existing_tokenizer:
        tokenizer = existing_tokenizer
    else:
        tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(labels_numeric)
    
    split_idx = int(len(X) * 0.8)
    if split_idx == 0:
        split_idx = len(X) - 1
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    log_info(request_id, f"Training model on {len(X_train)} samples (validation: {len(X_test)} samples)")
    
    model = build_model()
    
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=min(32, len(X_train)),
        validation_split=0.2 if len(X_train) > 10 else 0.0,
        verbose=1
    )
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0) if len(X_test) > 0 else (0.0, 0.0)
    if len(X_test) == 0:
        test_accuracy = float(history.history['accuracy'][-1])
    
    version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    log_info(request_id, f"Model training complete | New version: {version} | Accuracy: {test_accuracy:.4f}")
    
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    TOKENIZER_DIR.mkdir(exist_ok=True, parents=True)
    
    model_path = MODELS_DIR / f"model_{version}.keras"
    model.save(str(model_path))
    
    tokenizer_path = TOKENIZER_DIR / f"tokenizer_{version}.pkl"
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    model_registry_path = f"/models/model_{version}.keras"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            REGISTRY_URL,
            json={
                "version": version,
                "path": model_registry_path,
                "accuracy": float(test_accuracy),
                "is_active": True
            }
        )
        response.raise_for_status()
        
        await client.post(
            NOTIFICATION_URL,
            json={
                "event": "model_retrained",
                "details": {
                    "version": version,
                    "accuracy": float(test_accuracy),
                    "training_samples": len(feedback_list)
                }
            }
        )
    
    log_info(request_id, f"Model registered and notifications sent")
    
    return {
        "message": "Model retrained and registered",
        "version": version,
        "path": model_registry_path,
        "accuracy": float(test_accuracy),
        "training_samples": len(feedback_list)
    }

