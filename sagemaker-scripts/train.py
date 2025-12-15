"""
SageMaker Training Script for Sentiment Analysis
This script is executed by SageMaker Training Job
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Hyperparameters
MAX_SEQUENCE_LENGTH = 200
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128


def load_and_preprocess_data(data_dir):
    """Load and preprocess training data"""
    print(f"Loading data from {data_dir}")
    
    train_file = os.path.join(data_dir, 'train_data.csv')
    df = pd.read_csv(train_file)
    
    texts = df['reviews.text'].fillna('').astype(str).tolist()
    labels = df['sentiment'].fillna('Neutral').tolist()
    
    # Map labels to integers
    label_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
    labels_numeric = [label_map.get(label, 2) for label in labels]
    
    # Calculate class weights to handle imbalance
    from collections import Counter
    class_counts = Counter(labels_numeric)
    total_samples = len(labels_numeric)
    num_classes = len(class_counts)
    
    # Calculate class weights: n_samples / (n_classes * count_per_class)
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_samples / (num_classes * count)
    
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Class weights: {class_weights}")
    
    # Tokenize texts
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(labels_numeric)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, tokenizer, class_weights


def build_model():
    """Build sentiment analysis model with overfitting prevention"""
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    
    args, _ = parser.parse_known_args()
    
    print("Starting training...")
    print(f"Model dir: {args.model_dir}")
    print(f"Train data dir: {args.train}")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, tokenizer, class_weights = load_and_preprocess_data(args.train)
    
    # Build and train model
    print("Building model...")
    model = build_model()
    
    # Create proper train/val split with stratification
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training: {len(X_train_split)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Callbacks for overfitting prevention
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.0001,
        verbose=1
    )
    
    print("Training model with class weights...")
    history = model.fit(
        X_train_split, y_train_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Get final validation accuracy
    final_val_accuracy = None
    if 'val_accuracy' in history.history and len(history.history['val_accuracy']) > 0:
        final_val_accuracy = float(history.history['val_accuracy'][-1])
        print(f"Final validation accuracy: {final_val_accuracy:.4f}")
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Generate confusion matrix
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nValidation Accuracy: {final_val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save model in TensorFlow SavedModel format
    print(f"Saving model to {args.model_dir}")
    model_path = os.path.join(args.model_dir, '1')  # Version 1
    model.save(model_path, save_format='tf')
    
    # Save tokenizer as pickle (for SageMaker)
    tokenizer_path = os.path.join(args.model_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Also save tokenizer as JSON for inference service (version-independent)
    try:
        import boto3
        s3_client = boto3.client('s3')
        models_bucket = os.environ.get('MODELS_BUCKET') or os.environ.get('S3_MODELS_BUCKET')
        if models_bucket:
            # Save as JSON (version-independent)
            tokenizer_json = {
                'word_index': tokenizer.word_index,
                'num_words': tokenizer.num_words,
                'oov_token': tokenizer.oov_token,
                'document_count': tokenizer.document_count
            }
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(tokenizer_json, tmp_file)
                tmp_file_path = tmp_file.name
            
            s3_key = 'tokenizers/latest_tokenizer.json'
            s3_client.upload_file(tmp_file_path, models_bucket, s3_key)
            os.unlink(tmp_file_path)
            print(f"Tokenizer JSON saved to s3://{models_bucket}/{s3_key}")
            
            # Also save pickle for backward compatibility
            s3_key_pkl = 'tokenizers/latest_tokenizer.pkl'
            s3_client.upload_file(tokenizer_path, models_bucket, s3_key_pkl)
            print(f"Tokenizer pickle saved to s3://{models_bucket}/{s3_key_pkl}")
    except Exception as e:
        print(f"Warning: Could not save tokenizer to S3: {e}")
    
    # Save metrics
    metrics = {
        'accuracy': float(test_accuracy),
        'loss': float(test_loss),
        'validation_accuracy': float(final_val_accuracy) if final_val_accuracy is not None else None,
        'training_samples': len(X_train_split),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'confusion_matrix': cm.tolist()
    }
    
    metrics_path = os.path.join(args.model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    print("Training complete!")