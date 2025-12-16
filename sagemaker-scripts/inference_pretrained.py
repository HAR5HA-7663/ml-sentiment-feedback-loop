"""
SageMaker Inference Script for Pre-trained Product Review Sentiment Analyzer
Loads the TensorFlow model and performs sentiment predictions
Model: eakashyap/product-review-sentiment-analyzer (DistilBERT fine-tuned on Yelp reviews)
"""
import json
import os
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

# Global variables
model = None
tokenizer = None
reverse_label_map = None


def model_fn(model_dir):
    """Load model for inference"""
    global model, tokenizer, reverse_label_map

    print(f"Loading model from {model_dir}")

    # Load TensorFlow model
    model_path = os.path.join(model_dir, '1')
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")

    # Load tokenizer from pickle
    import pickle
    tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded from {tokenizer_path}")

    # Load reverse label mapping
    reverse_map_path = os.path.join(model_dir, 'reverse_label_map.json')
    with open(reverse_map_path, 'r') as f:
        reverse_label_map = json.load(f)
        # Convert string keys back to integers
        reverse_label_map = {int(k): v for k, v in reverse_label_map.items()}
    print(f"Label mapping loaded: {reverse_label_map}")

    print("Model, tokenizer, and label mapping loaded successfully!")
    return model


def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        data = json.loads(request_body)

        # Handle different input formats
        if 'instances' in data:
            texts = [item['text'] if isinstance(item, dict) else item for item in data['instances']]
        elif 'text' in data:
            texts = [data['text']]
        else:
            texts = [data] if isinstance(data, str) else [str(data)]

        return texts
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Make predictions"""
    global tokenizer, reverse_label_map

    # Tokenize
    encodings = tokenizer(
        input_data,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='tf'
    )

    # Predict
    predictions = model(encodings, training=False)
    logits = predictions['logits'] if isinstance(predictions, dict) else predictions
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()

    # Parse results
    results = []

    for pred in probabilities:
        predicted_class = int(np.argmax(pred))
        confidence = float(pred[predicted_class])

        # Map class to label using reverse mapping
        label = reverse_label_map.get(predicted_class, 'neutral').lower()

        results.append({
            'label': label,
            'confidence': confidence
        })

    return results


def output_fn(prediction, response_content_type):
    """Format output"""
    if response_content_type == 'application/json':
        return json.dumps({'predictions': prediction})
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
