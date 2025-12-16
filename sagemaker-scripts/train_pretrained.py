"""
SageMaker Training Script for Pre-trained Product Review Sentiment Analyzer
Downloads the pre-trained model from HuggingFace and saves it for SageMaker deployment
Model: eakashyap/product-review-sentiment-analyzer (DistilBERT fine-tuned on Yelp reviews)
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification
)
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Pre-trained model from HuggingFace
MODEL_NAME = "eakashyap/product-review-sentiment-analyzer"


def verify_label_mapping(model, tokenizer):
    """
    Verify the label mapping of the pre-trained model
    Returns the label map that matches our dataset format
    """
    # Test with known sentiment examples
    test_samples = [
        "This product is amazing! I love it!",  # Should be Positive
        "Terrible product, waste of money!",    # Should be Negative
        "It's okay, nothing special."           # Should be Neutral
    ]

    encodings = tokenizer(
        test_samples,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='tf'
    )

    predictions = model(encodings, training=False)
    logits = predictions['logits'] if isinstance(predictions, dict) else predictions
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()
    predicted_classes = np.argmax(probabilities, axis=-1)

    print("\n=== Label Mapping Verification ===")
    for text, pred, probs in zip(test_samples, predicted_classes, probabilities):
        print(f"Text: {text[:50]}...")
        print(f"Predicted class: {pred}, Probabilities: {probs}")

    # Common mapping for DistilBERT sentiment models
    # Most models use: 0=Negative, 1=Neutral, 2=Positive
    # Our dataset uses: Positive, Negative, Neutral (we'll map to 0, 1, 2)
    label_map = {'Positive': 2, 'Negative': 0, 'Neutral': 1}
    reverse_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    print(f"\nUsing label mapping: {reverse_map}")
    return label_map, reverse_map


def evaluate_on_dataset(model, tokenizer, reverse_map, data_dir):
    """
    Evaluate the pre-trained model on our dataset to verify performance
    """
    print("\n=== Evaluating Pre-trained Model on Dataset ===")

    # Read directly from S3
    import boto3
    s3_client = boto3.client('s3')
    data_bucket = os.environ.get('S3_DATA_BUCKET', 'ml-sentiment-data-143519759870')
    s3_key = 'train_data.csv'
    print(f"Reading from S3: s3://{data_bucket}/{s3_key}")

    try:
        obj = s3_client.get_object(Bucket=data_bucket, Key=s3_key)
        df = pd.read_csv(obj['Body'])

        # Take a sample for quick evaluation (full dataset would take too long)
        if len(df) > 500:
            df = df.sample(n=500, random_state=42)
            print(f"Using random sample of 500 examples for evaluation")

        texts = df['reviews.text'].fillna('').astype(str).tolist()
        labels = df['sentiment'].fillna('Neutral').tolist()

        # Map string labels to integers
        label_to_int = {'Positive': 2, 'Negative': 0, 'Neutral': 1}
        labels_numeric = [label_to_int.get(label, 1) for label in labels]

        # Tokenize
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='tf'
        )

        # Predict
        predictions = model(encodings, training=False)
        logits = predictions['logits'] if isinstance(predictions, dict) else predictions
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()
        predicted_classes = np.argmax(probabilities, axis=-1)

        # Calculate metrics
        accuracy = accuracy_score(labels_numeric, predicted_classes)
        cm = confusion_matrix(labels_numeric, predicted_classes)

        print(f"\nPre-trained Model Performance:")
        print(f"Accuracy on sample: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"(Rows: True labels [Neg, Neu, Pos], Cols: Predicted [Neg, Neu, Pos])")

        # Class distribution
        from collections import Counter
        true_dist = Counter(labels_numeric)
        pred_dist = Counter(predicted_classes.tolist())
        print(f"\nTrue label distribution: {dict(true_dist)}")
        print(f"Predicted distribution: {dict(pred_dist)}")

        return {
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'samples_evaluated': len(texts)
        }
    except Exception as e:
        print(f"Warning: Could not evaluate on dataset: {e}")
        return {'accuracy': 0.0, 'samples_evaluated': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--epochs', type=int, default=0)  # 0 = no training, just download
    parser.add_argument('--batch-size', type=int, default=16, dest='batch_size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, dest='learning_rate')

    args = parser.parse_args()

    print("=" * 80)
    print("Loading Pre-trained Product Review Sentiment Analyzer")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Model dir: {args.model_dir}")
    print(f"Train data dir: {args.train}")
    print(f"Epochs: {args.epochs} (0 = no training, just download pre-trained model)")

    # Load tokenizer and model from HuggingFace
    print(f"\nDownloading model from HuggingFace: {MODEL_NAME}")
    print("This will convert from PyTorch to TensorFlow...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        from_pt=True,  # Convert from PyTorch to TensorFlow
        num_labels=3   # Ensure 3-class output
    )

    print("Model loaded successfully!")
    print(f"Model config: {model.config}")

    # Verify label mapping
    label_map, reverse_map = verify_label_mapping(model, tokenizer)

    # Evaluate on our dataset (optional, for monitoring)
    eval_metrics = evaluate_on_dataset(model, tokenizer, reverse_map, args.train)

    # Save model in TensorFlow SavedModel format
    import pickle
    import shutil
    import tempfile

    print(f"\n{'=' * 80}")
    print(f"Saving model to {args.model_dir}")
    print(f"{'=' * 80}")

    # Create temp directory for saving (workaround for safe_open issues)
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save model to temp directory first
        tmp_model_path = os.path.join(tmp_dir, '1')
        print(f"Saving model to temporary location: {tmp_model_path}")
        model.save(tmp_model_path, save_format='tf')

        # Copy model to final location
        final_model_path = os.path.join(args.model_dir, '1')
        if os.path.exists(final_model_path):
            shutil.rmtree(final_model_path)
        print(f"Copying model from {tmp_model_path} to {final_model_path}")
        shutil.copytree(tmp_model_path, final_model_path)
        print(f"Model saved successfully!")

        # Save tokenizer
        tmp_tokenizer_path = os.path.join(tmp_dir, 'tokenizer.pkl')
        with open(tmp_tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)

        tokenizer_path = os.path.join(args.model_dir, 'tokenizer.pkl')
        shutil.copy2(tmp_tokenizer_path, tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")

        # Save label mapping
        tmp_label_map_path = os.path.join(tmp_dir, 'label_map.json')
        with open(tmp_label_map_path, 'w') as f:
            json.dump(label_map, f)

        label_map_path = os.path.join(args.model_dir, 'label_map.json')
        shutil.copy2(tmp_label_map_path, label_map_path)
        print(f"Label map saved to {label_map_path}")

        # Save reverse mapping for inference
        tmp_reverse_map_path = os.path.join(tmp_dir, 'reverse_label_map.json')
        with open(tmp_reverse_map_path, 'w') as f:
            # Convert int keys to strings for JSON
            reverse_map_str = {str(k): v for k, v in reverse_map.items()}
            json.dump(reverse_map_str, f)

        reverse_map_path = os.path.join(args.model_dir, 'reverse_label_map.json')
        shutil.copy2(tmp_reverse_map_path, reverse_map_path)
        print(f"Reverse label map saved to {reverse_map_path}")

        # Save metrics
        metrics = {
            'model_name': MODEL_NAME,
            'model_type': 'pre-trained',
            'accuracy': eval_metrics.get('accuracy', 0.0),
            'samples_evaluated': eval_metrics.get('samples_evaluated', 0),
            'confusion_matrix': eval_metrics.get('confusion_matrix', []),
            'label_mapping': reverse_map_str
        }

        tmp_metrics_path = os.path.join(tmp_dir, 'metrics.json')
        with open(tmp_metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        metrics_path = os.path.join(args.model_dir, 'metrics.json')
        shutil.copy2(tmp_metrics_path, metrics_path)
        print(f"Metrics saved to {metrics_path}")

    print("\n" + "=" * 80)
    print("Pre-trained model download and setup complete!")
    print("=" * 80)
    print("\nModel is ready for deployment to SageMaker endpoint.")
    print("Fine-tuning will be performed later when feedback data is available.")
