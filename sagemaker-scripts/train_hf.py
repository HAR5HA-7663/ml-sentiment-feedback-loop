"""
SageMaker Training Script for Sentiment Analysis using Hugging Face Transformers
Fine-tunes a pretrained model on feedback data
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch

# Use DistilBERT - smaller, faster, good for sentiment
MODEL_NAME = "distilbert-base-uncased"
# Alternative options:
# MODEL_NAME = "bert-base-uncased"  # Larger, better accuracy
# MODEL_NAME = "roberta-base"  # Better performance but larger


def load_and_preprocess_data(data_dir):
    """Load and preprocess training data"""
    print(f"Loading data from {data_dir}")
    
    train_file = os.path.join(data_dir, 'train_data.csv')
    df = pd.read_csv(train_file)
    
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


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    
    args, _ = parser.parse_known_args()
    
    print("Starting training with Hugging Face Transformers...")
    print(f"Model: {MODEL_NAME}")
    print(f"Model dir: {args.model_dir}")
    print(f"Train data dir: {args.train}")
    
    # Load and preprocess data
    texts, labels, label_map = load_and_preprocess_data(args.train)
    
    # Load tokenizer and model
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label={0: 'positive', 1: 'negative', 2: 'neutral'},
        label2id={'positive': 0, 'negative': 1, 'neutral': 2}
    )
    
    # Tokenize texts
    print("Tokenizing texts...")
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors=None
    )
    
    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })
    
    # Split data
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # Further split train into train/val
    train_val_split = train_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Training arguments (compatible with older transformers versions)
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f"{args.model_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        fp16=False,  # Disable fp16 for CPU compatibility
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test F1: {test_results['eval_f1']:.4f}")
    
    # Generate confusion matrix
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save model and tokenizer
    print(f"Saving model to {args.model_dir}")
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    
    # Save label mapping
    label_map_path = os.path.join(args.model_dir, 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f)
    
    # Save metrics
    metrics = {
        'accuracy': float(test_results['eval_accuracy']),
        'f1': float(test_results['eval_f1']),
        'test_samples': len(test_dataset),
        'confusion_matrix': cm.tolist()
    }
    
    metrics_path = os.path.join(args.model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    print("Training complete!")
