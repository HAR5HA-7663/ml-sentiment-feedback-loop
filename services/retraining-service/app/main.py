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
import os
import boto3
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from app.logging_middleware import RequestLoggingMiddleware, log_info

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)

FEEDBACK_FILE = Path("/feedback-data/feedback.json")
MODELS_DIR = Path("/models")
TOKENIZER_DIR = Path("/models/tokenizers")
REGISTRY_URL = "http://model-registry-service:8002/register"
NOTIFICATION_URL = "http://notification-service:8005/notify"

S3_BUCKET = os.getenv("S3_DATA_BUCKET", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

# Initialize S3 client
s3_client = boto3.client('s3', region_name=AWS_REGION) if S3_BUCKET else None

MAX_SEQUENCE_LENGTH = 200
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128

label_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
reverse_label_map = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}


@app.get("/health")
async def health():
    return {"status": "ok", "s3_enabled": bool(S3_BUCKET)}


def load_feedback_from_s3():
    """Load all feedback from S3"""
    if not s3_client or not S3_BUCKET:
        log_info("retraining", f"S3 client or bucket not available: client={s3_client is not None}, bucket={S3_BUCKET}")
        return []
    
    try:
        log_info("retraining", f"Loading feedback from S3 bucket: {S3_BUCKET}, prefix: feedback/")
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix='feedback/'
        )
        
        if 'Contents' not in response:
            log_info("retraining", "No feedback files found in S3 (Contents not in response)")
            return []
        
        log_info("retraining", f"Found {len(response['Contents'])} feedback files in S3")
        feedback_list = []
        for obj in response['Contents']:
            try:
                data = s3_client.get_object(Bucket=S3_BUCKET, Key=obj['Key'])
                content = json.loads(data['Body'].read().decode('utf-8'))
                feedback_list.append(content)
                log_info("retraining", f"Loaded feedback from {obj['Key']}")
            except Exception as e:
                log_info("retraining", f"Error reading {obj['Key']}: {e}")
                continue
        
        log_info("retraining", f"Successfully loaded {len(feedback_list)} feedback entries from S3")
        return feedback_list
    except Exception as e:
        log_info("retraining", f"Error loading feedback from S3: {e}")
        return []


def load_feedback():
    # Try S3 first, fall back to local file
    log_info("retraining", f"Loading feedback: S3_BUCKET={S3_BUCKET}, s3_client={s3_client is not None}")
    feedback = load_feedback_from_s3()
    if feedback:
        log_info("retraining", f"Loaded {len(feedback)} feedback entries from S3")
        return feedback
    
    log_info("retraining", "No feedback from S3, trying local file")
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, "r") as f:
            local_feedback = json.load(f)
            log_info("retraining", f"Loaded {len(local_feedback)} feedback entries from local file")
            return local_feedback
    
    log_info("retraining", "No feedback data found (neither S3 nor local file)")
    return []


def load_existing_tokenizer():
    tokenizer_files = sorted(TOKENIZER_DIR.glob("tokenizer_*.pkl"), reverse=True)
    if tokenizer_files:
        with open(tokenizer_files[0], 'rb') as f:
            return pickle.load(f)
    return None


def build_model():
    """Build model with overfitting prevention"""
    model = keras.Sequential([
        keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(3, activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
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
    
    # Proper train/val/test split
    if len(X) < 20:
        # Very small dataset - use simple split
        split_idx = int(len(X) * 0.8)
        if split_idx == 0:
            split_idx = len(X) - 1
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_val, y_val = X_test, y_test
    else:
        # Proper stratified split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
    
    log_info(request_id, f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    model = build_model()
    
    # Callbacks for overfitting prevention
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=0.0001,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=min(32, len(X_train)),
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Get validation accuracy from training history
    val_accuracy = None
    if 'val_accuracy' in history.history and len(history.history['val_accuracy']) > 0:
        val_accuracy = float(history.history['val_accuracy'][-1])
        log_info(request_id, f"Validation accuracy: {val_accuracy:.4f}")
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0) if len(X_test) > 0 else (0.0, 0.0)
    if len(X_test) == 0:
        test_accuracy = float(history.history['accuracy'][-1])
        cm = None
    else:
        # Generate confusion matrix on test set
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_test, y_pred_classes).tolist()
        log_info(request_id, f"Confusion matrix: {cm}")
    
    version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    log_info(request_id, f"Model training complete | New version: {version} | Test Accuracy: {test_accuracy:.4f}")
    
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
    
    result = {
        "message": "Model retrained and registered",
        "version": version,
        "path": model_registry_path,
        "test_accuracy": float(test_accuracy),
        "training_samples": len(feedback_list)
    }
    
    if val_accuracy is not None:
        result["validation_accuracy"] = val_accuracy
    
    if cm is not None:
        result["confusion_matrix"] = cm
        result["confusion_matrix_labels"] = ["Positive", "Negative", "Neutral"]
    
    return result

