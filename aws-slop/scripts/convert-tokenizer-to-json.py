#!/usr/bin/env python3
"""
Convert tokenizer pickle to JSON format
Run this with: python convert-tokenizer-to-json.py
Requires: tensorflow==2.11.0 (to match training environment)
"""
import pickle
import json
import boto3
import sys
import os

MODELS_BUCKET = "ml-sentiment-models-143519759870"
REGION = "us-east-2"

def main():
    s3 = boto3.client('s3', region_name=REGION)
    
    print("Downloading tokenizer pickle from S3...")
    pickle_path = "/tmp/tokenizer.pkl"
    s3.download_file(MODELS_BUCKET, "tokenizers/latest_tokenizer.pkl", pickle_path)
    
    print("Loading tokenizer...")
    with open(pickle_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    print("Extracting tokenizer data...")
    tokenizer_data = {
        'word_index': tokenizer.word_index,
        'num_words': getattr(tokenizer, 'num_words', 10000),
        'oov_token': getattr(tokenizer, 'oov_token', '<OOV>'),
        'document_count': getattr(tokenizer, 'document_count', 0)
    }
    
    print(f"Word index size: {len(tokenizer_data['word_index'])}")
    
    print("Saving as JSON...")
    json_path = "/tmp/tokenizer.json"
    with open(json_path, 'w') as f:
        json.dump(tokenizer_data, f)
    
    print("Uploading JSON to S3...")
    s3.upload_file(json_path, MODELS_BUCKET, "tokenizers/latest_tokenizer.json")
    
    print(f"✅ Tokenizer JSON uploaded to s3://{MODELS_BUCKET}/tokenizers/latest_tokenizer.json")
    
    # Cleanup
    os.unlink(pickle_path)
    os.unlink(json_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
