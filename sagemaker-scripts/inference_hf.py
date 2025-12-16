"""
SageMaker Inference Script for Sentiment Analysis using Hugging Face Transformers
"""
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Global variables
model = None
tokenizer = None
device = None


def model_fn(model_dir):
    """Load model for inference"""
    global model, tokenizer, device
    
    print(f"Loading model from {model_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    print("Model and tokenizer loaded successfully")
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
    global tokenizer, device
    
    # Tokenize
    encodings = tokenizer(
        input_data,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Move to device
    encodings = {k: v.to(device) for k, v in encodings.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Parse results
    results = []
    label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    
    predictions_cpu = predictions.cpu().numpy()
    for pred in predictions_cpu:
        predicted_class = int(pred.argmax())
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
