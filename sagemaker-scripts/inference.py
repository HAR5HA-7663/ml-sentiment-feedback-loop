"""
SageMaker Inference Script for Pre-trained Product Review Sentiment Analyzer
Uses PyTorch and HuggingFace transformers
"""
import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = None
tokenizer = None
reverse_label_map = None
device = None


def model_fn(model_dir):
    """Load model for inference"""
    global model, tokenizer, reverse_label_map, device

    print(f"Loading model from {model_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    print("Model loaded!")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("Tokenizer loaded!")

    # Load reverse label mapping
    reverse_map_path = os.path.join(model_dir, 'reverse_label_map.json')
    if os.path.exists(reverse_map_path):
        with open(reverse_map_path, 'r') as f:
            reverse_label_map = json.load(f)
            reverse_label_map = {int(k): v for k, v in reverse_label_map.items()}
    else:
        # Default mapping
        reverse_label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    print(f"Label mapping: {reverse_label_map}")
    return model


def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        data = json.loads(request_body)

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
    global tokenizer, reverse_label_map, device

    # Tokenize
    encodings = tokenizer(
        input_data,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

    # Parse results
    results = []
    for pred in probabilities:
        predicted_class = int(np.argmax(pred))
        confidence = float(pred[predicted_class])
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
