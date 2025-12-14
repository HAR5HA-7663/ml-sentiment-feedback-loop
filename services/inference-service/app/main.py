from fastapi import FastAPI, Request
from pydantic import BaseModel
import boto3
import json
import os
import pickle
import tempfile
import sys

# Import TensorFlow and Keras modules before unpickling tokenizer
# This ensures the tokenizer can be unpickled correctly
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app.logging_middleware import RequestLoggingMiddleware, log_info

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)

SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT_NAME", "ml-sentiment-endpoint")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
MODELS_BUCKET = os.getenv("S3_MODELS_BUCKET", "")
MAX_SEQUENCE_LENGTH = 200

# Initialize AWS clients
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION)

# Cache tokenizer
_tokenizer = None
_tokenizer_loaded = False


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    label: str
    confidence: float


@app.get("/health")
async def health():
    return {"status": "ok", "sagemaker_endpoint": SAGEMAKER_ENDPOINT}


def load_tokenizer():
    """Load tokenizer from S3 (cached after first load)"""
    global _tokenizer, _tokenizer_loaded
    
    if _tokenizer_loaded:
        return _tokenizer
    
    if not MODELS_BUCKET:
        log_info("inference", "S3_MODELS_BUCKET not set, cannot load tokenizer")
        return None
    
    try:
        # Find the latest training job output
        # Look for tokenizer in the most recent training output
        log_info("inference", f"Loading tokenizer from s3://{MODELS_BUCKET}")
        
        # Try to find tokenizer in training outputs
        # Format: s3://bucket/training-output/job-name/output/model.tar.gz
        # Tokenizer is inside model.tar.gz at tokenizer.pkl
        # For now, try to get it from the latest training job
        
        # List training outputs
        prefix = "training-output/"
        response = s3_client.list_objects_v2(Bucket=MODELS_BUCKET, Prefix=prefix, Delimiter='/')
        
        if 'CommonPrefixes' not in response:
            log_info("inference", "No training outputs found")
            return None
        
        # Get the most recent training job
        training_jobs = sorted([p['Prefix'] for p in response['CommonPrefixes']], reverse=True)
        if not training_jobs:
            log_info("inference", "No training jobs found")
            return None
        
        # Try to download tokenizer from the model artifacts
        # The tokenizer is in the model.tar.gz which SageMaker extracts
        # We need to download the model.tar.gz, extract it, and get tokenizer.pkl
        # But this is complex - let's try a simpler approach: store tokenizer separately
        
        # Try to load tokenizer from JSON (version-independent)
        tokenizer_key_json = "tokenizers/latest_tokenizer.json"
        try:
            # Download JSON file
            tmp_file_path = tempfile.mktemp(suffix='.json')
            s3_client.download_file(MODELS_BUCKET, tokenizer_key_json, tmp_file_path)
            
            # Load JSON and recreate Tokenizer
            with open(tmp_file_path, 'r') as f:
                tokenizer_data = json.load(f)
            
            # Recreate Tokenizer object
            _tokenizer = Tokenizer(
                num_words=tokenizer_data.get('num_words', 10000),
                oov_token=tokenizer_data.get('oov_token', '<OOV>')
            )
            _tokenizer.word_index = tokenizer_data['word_index']
            _tokenizer.document_count = tokenizer_data.get('document_count', 0)
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            _tokenizer_loaded = True
            log_info("inference", "Tokenizer loaded successfully from JSON")
            return _tokenizer
        except Exception as e:
            log_info("inference", f"Error loading tokenizer JSON: {str(e)}")
            # Fallback to pickle (for backward compatibility)
            tokenizer_key_pkl = "tokenizers/latest_tokenizer.pkl"
            try:
                tmp_file_path = tempfile.mktemp(suffix='.pkl')
                s3_client.download_file(MODELS_BUCKET, tokenizer_key_pkl, tmp_file_path)
                with open(tmp_file_path, 'rb') as f:
                    _tokenizer = pickle.load(f)
                os.unlink(tmp_file_path)
                _tokenizer_loaded = True
                log_info("inference", "Tokenizer loaded from pickle (fallback)")
                return _tokenizer
            except Exception as e2:
                log_info("inference", f"Error loading tokenizer pickle: {str(e2)}")
                return None
            
    except Exception as e:
        log_info("inference", f"Error loading tokenizer: {e}")
        return None


async def predict_sentiment(text: str) -> tuple[str, float]:
    """Call SageMaker endpoint for prediction using TensorFlow Serving format"""
    try:
        # Load tokenizer
        tokenizer = load_tokenizer()
        if not tokenizer:
            log_info("inference", "Tokenizer not available, cannot preprocess text")
            return "neutral", 0.5
        
        # Preprocess text: tokenize and pad
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH).tolist()
        
        # Prepare payload for TensorFlow Serving
        # TensorFlow Serving expects: {"instances": [[...sequence...]]}
        payload = {
            "instances": padded
        }
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        result = json.loads(response['Body'].read().decode())
        
        # Parse TensorFlow Serving response
        # Format: {"predictions": [[prob_class0, prob_class1, prob_class2], ...]}
        if 'predictions' in result and len(result['predictions']) > 0:
            predictions = result['predictions'][0]  # Get first prediction
            predicted_class = int(predictions.index(max(predictions)))
            confidence = float(max(predictions))
            
            # Map to labels (0=positive, 1=negative, 2=neutral)
            label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
            label = label_map.get(predicted_class, 'neutral')
            
            return label, confidence
        else:
            log_info("inference", f"Unexpected response format: {result}")
            return "neutral", 0.5
        
    except Exception as e:
        log_info("inference", f"SageMaker error: {e}, falling back to neutral")
        # Fallback to neutral if SageMaker fails
        return "neutral", 0.5


@app.post("/predict-sentiment", response_model=SentimentResponse)
async def predict_sentiment_endpoint(request: Request, req_body: SentimentRequest):
    request_id = getattr(request.state, "request_id", "unknown")
    
    log_info(request_id, f"Predicting sentiment for text (length: {len(req_body.text)} chars)")
    log_info(request_id, f"Using SageMaker endpoint: {SAGEMAKER_ENDPOINT}")
    
    label, confidence = await predict_sentiment(req_body.text)
    
    log_info(request_id, f"Prediction: {label} (confidence: {confidence:.4f})")
    
    return SentimentResponse(label=label, confidence=confidence)


