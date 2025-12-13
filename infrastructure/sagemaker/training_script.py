"""
SageMaker Training Script for Sentiment Analysis
Uses TensorFlow to train a sentiment classifier on Amazon product reviews
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
import pickle

# Hyperparameters
MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128
LSTM_UNITS = 64

def load_and_preprocess_data(data_dir):
    """Load and preprocess the training data"""
    print("Loading training data...")
    train_file = os.path.join(data_dir, 'train_data.csv')
    df = pd.read_csv(train_file)
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract text and labels
    texts = df['reviews.text'].fillna('').astype(str).tolist()
    labels = df['sentiment'].str.lower().tolist()
    
    # Convert labels to binary (Positive=1, Negative/Neutral=0)
    label_map = {'positive': 1, 'negative': 0, 'neutral': 0}
    y = np.array([label_map.get(label, 0) for label in labels])
    
    print(f"Positive samples: {sum(y)}, Negative/Neutral samples: {len(y) - sum(y)}")
    
    # Tokenize text
    print("Tokenizing text...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Sequence shape: {X.shape}")
    
    return X, y, tokenizer

def build_model():
    """Build the sentiment analysis model"""
    print("Building model...")
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(LSTM_UNITS //2)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    print("Model architecture:")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    """Train the model"""
    print(f"Training model for {epochs} epochs...")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def save_model(model, tokenizer, model_dir):
    """Save the model and tokenizer"""
    print(f"Saving model to {model_dir}...")
    
    # Save model in SavedModel format
    model_path = os.path.join(model_dir, 'model', '1')
    model.save(model_path, save_format='tf')
    
    # Save tokenizer
    tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Save model config
    config = {
        'max_words': MAX_WORDS,
        'max_sequence_length': MAX_SEQUENCE_LENGTH,
        'embedding_dim': EMBEDDING_DIM,
        'lstm_units': LSTM_UNITS
    }
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    print("Model saved successfully!")

def main():
    parser = argparse.ArgumentParser()
    
    # SageMaker environment variables
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    
    args = parser.parse_args()
    
    print("="*50)
    print("SageMaker Sentiment Analysis Training")
    print("="*50)
    print(f"Model directory: {args.model_dir}")
    print(f"Training data directory: {args.train}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*50)
    
    # Load and preprocess data
    X, y, tokenizer = load_and_preprocess_data(args.train)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build model
    model = build_model()
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val, args.epochs, args.batch_size)
    
    # Evaluate on validation set
    print("\nFinal evaluation on validation set:")
    val_loss, val_acc, val_prec, val_rec = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Precision: {val_prec:.4f}")
    print(f"Validation Recall: {val_rec:.4f}")
    
    # Save model and tokenizer
    save_model(model, tokenizer, args.model_dir)
    
    print("\nTraining completed successfully!")

if __name__ == '__main__':
    main()
