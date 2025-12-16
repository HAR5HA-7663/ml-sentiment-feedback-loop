"""
SageMaker Training Script for Sentiment Analysis using TensorFlow + Hugging Face Transformers
Fine-tunes a pretrained DistilBERT model on feedback data using TensorFlow
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import (
    AutoTokenizer, 
    TFDistilBertForSequenceClassification
)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Use DistilBERT - smaller, faster, good for sentiment
MODEL_NAME = "distilbert-base-uncased"


def load_and_preprocess_data(data_dir):
    """Load and preprocess training data"""
    print(f"Loading data from S3 (bypassing SageMaker file system issue)...")
    
    # Read directly from S3 - more reliable than SageMaker file system
    import boto3
    s3_client = boto3.client('s3')
    data_bucket = os.environ.get('S3_DATA_BUCKET', 'ml-sentiment-data-143519759870')
    s3_key = 'train_data.csv'
    print(f"Reading from S3: s3://{data_bucket}/{s3_key}")
    obj = s3_client.get_object(Bucket=data_bucket, Key=s3_key)
    df = pd.read_csv(obj['Body'])
    
    texts = df['reviews.text'].fillna('').astype(str).tolist()
    labels = df['sentiment'].fillna('Neutral').tolist()
    
    # Map labels to integers (matching Hugging Face format)
    label_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
    labels_numeric = [label_map.get(label, 2) for label in labels]
    
    # Handle class imbalance with undersampling
    from collections import Counter
    np.random.seed(42)
    
    class_counts = Counter(labels_numeric)
    print(f"Original class distribution: {dict(class_counts)}")
    
    min_class_size = min(class_counts.values())
    print(f"Minority class size: {min_class_size}")
    
    # Undersample to 1:1 balance
    texts_array = np.array(texts)
    labels_array = np.array(labels_numeric)
    
    balanced_texts = []
    balanced_labels = []
    
    for class_idx in range(3):
        class_mask = labels_array == class_idx
        class_texts = texts_array[class_mask].tolist()
        class_labels = labels_array[class_mask].tolist()
        
        if class_idx == 0:  # Positive - undersample
            if len(class_texts) > min_class_size:
                indices = np.random.choice(len(class_texts), size=min_class_size, replace=False)
                class_texts = [class_texts[i] for i in indices]
                class_labels = [class_labels[i] for i in indices]
                print(f"Undersampled Positive: {len(class_texts)} samples")
        else:
            print(f"Keeping all {['Positive', 'Negative', 'Neutral'][class_idx]}: {len(class_texts)} samples")
        
        balanced_texts.extend(class_texts)
        balanced_labels.extend(class_labels)
    
    # Shuffle
    indices = np.random.permutation(len(balanced_texts))
    balanced_texts = [balanced_texts[i] for i in indices]
    balanced_labels = [balanced_labels[i] for i in indices]
    
    balanced_counts = Counter(balanced_labels)
    print(f"Balanced class distribution: {dict(balanced_counts)}")
    
    return balanced_texts, balanced_labels, label_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    # SageMaker passes hyperparameters with hyphens, argparse converts to underscores
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16, dest='batch_size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, dest='learning_rate')
    
    args = parser.parse_args()
    
    # Ensure batch_size and learning_rate are set (argparse converts hyphens to underscores)
    if not hasattr(args, 'batch_size') or args.batch_size is None:
        args.batch_size = 16
    if not hasattr(args, 'learning_rate') or args.learning_rate is None:
        args.learning_rate = 2e-5
    
    print("Starting training with TensorFlow + Hugging Face Transformers...")
    print(f"Model: {MODEL_NAME} (TensorFlow)")
    print(f"Model dir: {args.model_dir}")
    print(f"Train data dir: {args.train}")
    
    # Load and preprocess data
    texts, labels, label_map = load_and_preprocess_data(args.train)
    
    # Load tokenizer and model (TensorFlow version)
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TFDistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )
    
    # Split data first, then tokenize
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(texts))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=[labels[i] for i in temp_idx])
    
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Tokenize splits (return as lists, not tensors)
    print("Tokenizing texts...")
    train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=512, return_tensors=None)
    val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=512, return_tensors=None)
    test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', max_length=512, return_tensors=None)
    
    # Convert to numpy arrays (handle list of lists)
    train_input_ids = np.array(train_encodings['input_ids'], dtype=np.int32)
    train_attention_mask = np.array(train_encodings['attention_mask'], dtype=np.int32)
    train_labels_array = np.array(train_labels, dtype=np.int32)
    
    val_input_ids = np.array(val_encodings['input_ids'], dtype=np.int32)
    val_attention_mask = np.array(val_encodings['attention_mask'], dtype=np.int32)
    val_labels_array = np.array(val_labels, dtype=np.int32)
    
    test_input_ids = np.array(test_encodings['input_ids'], dtype=np.int32)
    test_attention_mask = np.array(test_encodings['attention_mask'], dtype=np.int32)
    test_labels_array = np.array(test_labels, dtype=np.int32)
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': train_input_ids, 'attention_mask': train_attention_mask},
        train_labels_array
    )).batch(args.batch_size)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': val_input_ids, 'attention_mask': val_attention_mask},
        val_labels_array
    )).batch(args.batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': test_input_ids, 'attention_mask': test_attention_mask},
        test_labels_array
    )).batch(args.batch_size)
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train
    print("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = model.evaluate(test_dataset, verbose=0)
    test_loss = test_results[0]
    test_accuracy = test_results[1]
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Generate confusion matrix
    test_predictions = model.predict(test_dataset, verbose=0)
    y_pred = np.argmax(test_predictions, axis=1)
    y_true = np.array(test_labels)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save model in TensorFlow SavedModel format
    # Workaround: Save to /tmp first, then copy to SM_MODEL_DIR to avoid safe_open issues
    import pickle
    import shutil
    import tempfile
    
    print(f"Saving model to {args.model_dir}")
    
    # Create temp directory for saving
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save model to temp directory first
        tmp_model_path = os.path.join(tmp_dir, '1')
        print(f"Saving model to temporary location: {tmp_model_path}")
        model.save(tmp_model_path, save_format='tf')
        
        # Copy model to final location
        final_model_path = os.path.join(args.model_dir, '1')
        # Remove existing directory if it exists
        if os.path.exists(final_model_path):
            shutil.rmtree(final_model_path)
        print(f"Copying model from {tmp_model_path} to {final_model_path}")
        if os.path.exists(tmp_model_path):
            shutil.copytree(tmp_model_path, final_model_path)
        print(f"Model saved successfully to {final_model_path}")
        
        # Save tokenizer to temp file first, then copy
        tmp_tokenizer_path = os.path.join(tmp_dir, 'tokenizer.pkl')
        with open(tmp_tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        
        tokenizer_path = os.path.join(args.model_dir, 'tokenizer.pkl')
        shutil.copy2(tmp_tokenizer_path, tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
        
        # Save label mapping to temp file first, then copy
        tmp_label_map_path = os.path.join(tmp_dir, 'label_map.json')
        with open(tmp_label_map_path, 'w') as f:
            json.dump(label_map, f)
        
        label_map_path = os.path.join(args.model_dir, 'label_map.json')
        shutil.copy2(tmp_label_map_path, label_map_path)
        print(f"Label map saved to {label_map_path}")
        
        # Save metrics to temp file first, then copy
        metrics = {
            'accuracy': float(test_accuracy),
            'loss': float(test_loss),
            'test_samples': len(test_texts),
            'confusion_matrix': cm.tolist()
        }
        
        tmp_metrics_path = os.path.join(tmp_dir, 'metrics.json')
        with open(tmp_metrics_path, 'w') as f:
            json.dump(metrics, f)
        
        metrics_path = os.path.join(args.model_dir, 'metrics.json')
        shutil.copy2(tmp_metrics_path, metrics_path)
        print(f"Metrics saved to {metrics_path}")
    
    print("Training complete!")
