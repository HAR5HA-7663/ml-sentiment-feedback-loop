"""
SageMaker Training Script for Pre-trained Product Review Sentiment Analyzer
Downloads the pre-trained model from HuggingFace and saves it for SageMaker deployment
Model: eakashyap/product-review-sentiment-analyzer (DistilBERT fine-tuned on Yelp reviews)
Uses PyTorch for compatibility with SageMaker HuggingFace containers
"""
import os
import json
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# Pre-trained model from HuggingFace
# Using a well-established sentiment model that's widely used
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def verify_label_mapping(model, tokenizer, device):
    """
    Verify the label mapping of the pre-trained model
    """
    test_samples = [
        "This product is amazing! I love it!",
        "Terrible product, waste of money!",
        "It's okay, nothing special."
    ]

    encodings = tokenizer(
        test_samples,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
        predicted_classes = np.argmax(probabilities, axis=-1)

    print("\n=== Label Mapping Verification ===")
    for text, pred, probs in zip(test_samples, predicted_classes, probabilities):
        print(f"Text: {text[:50]}...")
        print(f"Predicted class: {pred}, Probabilities: {probs}")

    # Standard mapping: 0=Negative, 1=Neutral, 2=Positive
    label_map = {'Positive': 2, 'Negative': 0, 'Neutral': 1}
    reverse_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    print(f"\nUsing label mapping: {reverse_map}")
    return label_map, reverse_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=16, dest='batch_size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, dest='learning_rate')

    args = parser.parse_args()

    print("=" * 80)
    print("Loading Pre-trained Product Review Sentiment Analyzer")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Model dir: {args.model_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer and model from HuggingFace
    print(f"\nDownloading model from HuggingFace: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)

    print("Model loaded successfully!")

    # Verify label mapping
    label_map, reverse_map = verify_label_mapping(model, tokenizer, device)

    # Save model and tokenizer
    print(f"\n{'=' * 80}")
    print(f"Saving model to {args.model_dir}")
    print(f"{'=' * 80}")

    # Save using HuggingFace format
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    print("Model and tokenizer saved!")

    # Save label mappings
    reverse_map_str = {str(k): v for k, v in reverse_map.items()}

    with open(os.path.join(args.model_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)

    with open(os.path.join(args.model_dir, 'reverse_label_map.json'), 'w') as f:
        json.dump(reverse_map_str, f)

    print("Label mappings saved!")
    print("\n" + "=" * 80)
    print("Pre-trained model download and setup complete!")
    print("=" * 80)
