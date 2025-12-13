from fastapi import FastAPI, Request
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json
import httpx
from pathlib import Path
from datetime import datetime
import numpy as np
from app.logging_middleware import RequestLoggingMiddleware, log_info

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)

DATASET_FILE = Path("/dataset/train_data.csv")
MODELS_DIR = Path("/models")
TOKENIZER_DIR = Path("/models/tokenizers")
REGISTRY_URL = "http://model-registry-service:8002/register"

MAX_SEQUENCE_LENGTH = 200
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128


def load_and_preprocess_data():
    df = pd.read_csv(DATASET_FILE)
    
    texts = df['reviews.text'].fillna('').astype(str).tolist()
    labels = df['sentiment'].fillna('Neutral').tolist()
    
    label_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
    labels_numeric = [label_map.get(label, 2) for label in labels]
    
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(labels_numeric)
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, tokenizer


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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/bootstrap")
async def bootstrap(request: Request):
    request_id = getattr(request.state, "request_id", "unknown")
    
    try:
        log_info(request_id, "Starting initial model bootstrap")
        
        X_train, X_test, y_train, y_test, tokenizer = load_and_preprocess_data()
        
        log_info(request_id, f"Data loaded: {len(X_train)} training samples, {len(X_test)} test samples")
        
        log_info(request_id, "Building and training initial model")
        
        model = build_model()
        
        history = model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        log_info(request_id, f"Model training complete | Test accuracy: {test_accuracy:.4f}")
        
        version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
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
        
        log_info(request_id, f"Bootstrap complete | Model version: {version} | Registered as active")
        
        return {
            "message": "Model trained and registered",
            "version": version,
            "path": model_registry_path,
            "accuracy": float(test_accuracy),
            "test_loss": float(test_loss)
        }
    
    except Exception as e:
        return {"error": str(e)}

