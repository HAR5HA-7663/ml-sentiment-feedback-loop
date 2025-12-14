"""
SageMaker Inference Script for Sentiment Analysis
This script handles prediction requests to the SageMaker endpoint
"""
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

MAX_SEQUENCE_LENGTH = 200

# Global variables to hold model and tokenizer
model = None
tokenizer = None


def model_fn(model_dir):
    """Load model for inference"""
    global model, tokenizer
    
    print(f"Loading model from {model_dir}")
    
    # Load TensorFlow model
    model_path = os.path.join(model_dir, '1')
    model = tf.keras.models.load_model(model_path)
    
    # Load tokenizer
    tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    print("Model and tokenizer loaded successfully")
    return model


def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        
        # Handle different input formats
        if 'instances' in data:
            texts = [item['text'] for item in data['instances']]
        elif 'text' in data:
            texts = [data['text']]
        else:
            texts = [data]
        
        return texts
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Make predictions"""
    global tokenizer
    
    # Tokenize and pad sequences
    sequences = tokenizer.texts_to_sequences(input_data)
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Get predictions
    predictions = model.predict(padded, verbose=0)
    
    # Parse results
    results = []
    label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    
    for pred in predictions:
        predicted_class = int(np.argmax(pred))
        confidence = float(pred[predicted_class])
        
        results.append({
            'label': label_map.get(predicted_class, 'neutral'),
            'confidence': confidence
        })
    
    return results


def output_fn(prediction, response_content_type):
    """Format output"""
    if response_content_type == 'application/json':
        return json.dumps({'predictions': prediction})
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
