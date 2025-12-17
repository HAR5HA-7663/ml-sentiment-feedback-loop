"""
SageMaker Training Script for Fine-tuning Sentiment Model on Feedback Data
Fine-tunes cardiffnlp/twitter-roberta-base-sentiment-latest on user feedback
"""
import os
import json
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Pre-trained model from HuggingFace
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class SentimentDataset(Dataset):
    """Custom dataset for sentiment classification"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_training_data(data_dir):
    """Load training data from CSV file"""
    print(f"Looking for training data in: {data_dir}")

    # Find CSV file
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    csv_path = os.path.join(data_dir, csv_files[0])
    print(f"Loading data from: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head()}")

    # Expected columns: text, label
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"CSV must have 'text' and 'label' columns. Found: {df.columns.tolist()}")

    texts = df['text'].tolist()
    labels = df['label'].astype(int).tolist()

    # Print label distribution
    from collections import Counter
    label_counts = Counter(labels)
    print(f"Label distribution: {dict(label_counts)}")

    return texts, labels


def train_model(model, train_loader, optimizer, scheduler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_model(model, data_loader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--training', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8, dest='batch_size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, dest='learning_rate')

    args = parser.parse_args()

    print("=" * 80)
    print("Fine-tuning Sentiment Model on Feedback Data")
    print("=" * 80)
    print(f"Base model: {MODEL_NAME}")
    print(f"Training data: {args.training}")
    print(f"Model output: {args.model_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load training data
    print("\n--- Loading Training Data ---")
    texts, labels = load_training_data(args.training)

    # Load tokenizer and model
    print(f"\n--- Loading Pre-trained Model: {MODEL_NAME} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device)
    print("Model loaded successfully!")

    # Create dataset and dataloader
    print("\n--- Preparing Dataset ---")
    dataset = SentimentDataset(texts, labels, tokenizer, max_length=128)

    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Training loop
    print("\n--- Starting Training ---")
    best_val_accuracy = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_model(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        if len(val_loader) > 0:
            val_loss, val_acc = evaluate_model(model, val_loader, device)
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                print(f"New best validation accuracy: {val_acc:.4f}")

    # Save model
    print(f"\n--- Saving Model to {args.model_dir} ---")
    os.makedirs(args.model_dir, exist_ok=True)

    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    # Save label mappings
    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    reverse_map = {'0': 'Negative', '1': 'Neutral', '2': 'Positive'}

    with open(os.path.join(args.model_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)

    with open(os.path.join(args.model_dir, 'reverse_label_map.json'), 'w') as f:
        json.dump(reverse_map, f)

    print("Model and tokenizer saved!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print("\n" + "=" * 80)
    print("Fine-tuning complete!")
    print("=" * 80)
