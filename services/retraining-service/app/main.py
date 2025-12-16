from fastapi import FastAPI, Request
import json
import httpx
import tensorflow as tf
from tensorflow import keras
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import os
import boto3
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
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

# Pre-trained model configuration
MODEL_NAME = "eakashyap/product-review-sentiment-analyzer"

# Label mapping (matching the pre-trained model)
# Pre-trained model uses: 0=Negative, 1=Neutral, 2=Positive
label_map = {'Positive': 2, 'Negative': 0, 'Neutral': 1}
reverse_label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}


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


def load_pretrained_model_and_tokenizer():
    """Load pre-trained model and tokenizer from HuggingFace"""
    log_info("retraining", f"Loading pre-trained model: {MODEL_NAME}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Load model (convert from PyTorch to TensorFlow)
        model = TFAutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            from_pt=True,
            num_labels=3
        )

        log_info("retraining", "Pre-trained model and tokenizer loaded successfully")
        return model, tokenizer
    except Exception as e:
        log_info("retraining", f"Error loading pre-trained model: {e}")
        raise


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

    labels_numeric = [label_map.get(label, 1) for label in labels]  # Default to Neutral (1)

    # Load pre-trained model and tokenizer
    model, tokenizer = load_pretrained_model_and_tokenizer()

    # Tokenize texts using HuggingFace tokenizer
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors=None  # Return as lists
    )

    # Convert to numpy arrays
    input_ids = np.array(encodings['input_ids'], dtype=np.int32)
    attention_mask = np.array(encodings['attention_mask'], dtype=np.int32)
    y = np.array(labels_numeric, dtype=np.int32)
    
    # Calculate class weights for imbalanced data (stronger weights for extreme imbalance)
    from collections import Counter
    class_counts = Counter(y.tolist())
    total_samples = len(y)
    num_classes = len(class_counts)
    
    # Use stronger class weights for extreme imbalance
    class_weights = {}
    for class_idx, count in class_counts.items():
        # Stronger weighting: (total_samples / count) ** 1.5
        base_weight = (total_samples / count) ** 1.5
        class_weights[class_idx] = base_weight
    
    # Normalize weights
    max_weight = max(class_weights.values())
    class_weights = {k: v / max_weight * 10 for k, v in class_weights.items()}
    
    log_info(request_id, f"Class distribution: {dict(class_counts)}")
    log_info(request_id, f"Class weights (normalized): {class_weights}")
    
    # Proper train/val/test split
    indices = np.arange(len(input_ids))

    if len(indices) < 20:
        # Very small dataset - use simple split
        split_idx = int(len(indices) * 0.8)
        if split_idx == 0:
            split_idx = len(indices) - 1
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        val_idx = test_idx
    else:
        # Proper stratified split
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.3, random_state=42, stratify=y
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42,
            stratify=y[temp_idx]
        )

    # Create train/val/test datasets
    train_input_ids = input_ids[train_idx]
    train_attention_mask = attention_mask[train_idx]
    train_labels = y[train_idx]

    val_input_ids = input_ids[val_idx]
    val_attention_mask = attention_mask[val_idx]
    val_labels = y[val_idx]

    test_input_ids = input_ids[test_idx]
    test_attention_mask = attention_mask[test_idx]
    test_labels = y[test_idx]

    log_info(request_id, f"Training: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}")

    # Create TensorFlow datasets
    batch_size = min(8, len(train_idx))  # Smaller batch size for fine-tuning

    train_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': train_input_ids, 'attention_mask': train_attention_mask},
        train_labels
    )).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': val_input_ids, 'attention_mask': val_attention_mask},
        val_labels
    )).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': test_input_ids, 'attention_mask': test_attention_mask},
        test_labels
    )).batch(batch_size)
    
    # Compile model for fine-tuning
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)  # Lower LR for fine-tuning
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    log_info(request_id, "Model compiled for fine-tuning")

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
        min_lr=1e-6,
        verbose=1
    )

    # Fine-tune the model
    log_info(request_id, "Starting fine-tuning...")
    history = model.fit(
        train_dataset,
        epochs=5,  # Fewer epochs for fine-tuning
        validation_data=val_dataset,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Get validation accuracy from training history
    val_accuracy = None
    if 'val_accuracy' in history.history and len(history.history['val_accuracy']) > 0:
        val_accuracy = float(history.history['val_accuracy'][-1])
        log_info(request_id, f"Validation accuracy: {val_accuracy:.4f}")
    
    # Evaluate on test set
    if len(test_idx) > 0:
        test_results = model.evaluate(test_dataset, verbose=0)
        test_loss = test_results[0]
        test_accuracy = test_results[1]

        # Generate confusion matrix on test set
        test_predictions = model.predict(test_dataset, verbose=0)
        # Handle HuggingFace model output format
        if hasattr(test_predictions, 'logits'):
            logits = test_predictions.logits
        else:
            logits = test_predictions

        y_pred_classes = np.argmax(logits, axis=1)
        cm = confusion_matrix(test_labels, y_pred_classes).tolist()
        log_info(request_id, f"Confusion matrix: {cm}")
    else:
        test_accuracy = float(history.history['accuracy'][-1])
        test_loss = 0.0
        cm = None
    
    version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    log_info(request_id, f"Model training complete | New version: {version} | Test Accuracy: {test_accuracy:.4f}")
    
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    TOKENIZER_DIR.mkdir(exist_ok=True, parents=True)

    # Save model in TensorFlow SavedModel format (compatible with HuggingFace)
    model_path = MODELS_DIR / f"model_{version}"
    model.save_pretrained(str(model_path))
    log_info(request_id, f"Model saved to {model_path}")

    # Save tokenizer
    tokenizer_path = TOKENIZER_DIR / f"tokenizer_{version}.pkl"
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    log_info(request_id, f"Tokenizer saved to {tokenizer_path}")

    # Save reverse label mapping
    label_map_path = MODELS_DIR / f"reverse_label_map_{version}.json"
    with open(label_map_path, 'w') as f:
        reverse_map_str = {str(k): v for k, v in reverse_label_map.items()}
        json.dump(reverse_map_str, f)
    log_info(request_id, f"Label mapping saved to {label_map_path}")

    model_registry_path = f"/models/model_{version}"
    
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
        result["confusion_matrix_labels"] = ["Negative", "Neutral", "Positive"]  # Matches label mapping: 0=Neg, 1=Neu, 2=Pos
    
    return result

