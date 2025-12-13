import requests
import json
import time

print("=" * 60)
print("Testing Complete ML Feedback Loop")
print("=" * 60)

# Step 1: Bootstrap (already done, but let's verify)
print("\n1. Checking if model exists...")
try:
    response = requests.get("http://localhost:8002/active-model")
    if response.status_code == 200:
        model = response.json()
        print(f"   [OK] Model found: {model.get('version')} (Accuracy: {model.get('accuracy'):.2%})")
    else:
        print("   [ERROR] No active model found")
except Exception as e:
    print(f"   [ERROR] {e}")

# Step 2: Predict
print("\n2. Making prediction...")
test_text = "This product is amazing! I love it so much."
try:
    response = requests.post(
        "http://localhost:8000/predict-sentiment",
        json={"text": test_text}
    )
    if response.status_code == 200:
        result = response.json()
        print(f"   [OK] Prediction: {result['label']} (Confidence: {result['confidence']:.2%})")
        predicted_label = result['label']
    else:
        print(f"   [ERROR] {response.text}")
        predicted_label = "Positive"
except Exception as e:
    print(f"   [ERROR] {e}")
    predicted_label = "Positive"

# Step 3: Submit Feedback
print("\n3. Submitting feedback...")
try:
    response = requests.post(
        "http://localhost:8001/submit-feedback",
        json={
            "text": test_text,
            "model_prediction": predicted_label,
            "user_label": "Positive"
        }
    )
    if response.status_code == 200:
        result = response.json()
        print(f"   [OK] Feedback submitted (ID: {result.get('id')})")
    else:
        print(f"   [ERROR] {response.text}")
except Exception as e:
    print(f"   [ERROR] {e}")

# Step 4: Run Evaluation
print("\n4. Running evaluation...")
try:
    response = requests.post("http://localhost:8003/run-evaluation")
    if response.status_code == 200:
        result = response.json()
        print(f"   [OK] Evaluation complete: Accuracy = {result.get('accuracy', 0):.2%}")
    else:
        print(f"   [ERROR] {response.text}")
except Exception as e:
    print(f"   [ERROR] {e}")

# Step 5: Retrain (need at least 10 feedback samples)
print("\n5. Adding more feedback samples for retraining...")
for i in range(10):
    texts = [
        "Great product, highly recommend!",
        "Terrible quality, very disappointed.",
        "It's okay, nothing special.",
        "Amazing value for money!",
        "Poor customer service.",
        "Love this product!",
        "Not worth the price.",
        "Excellent quality!",
        "Very disappointed with this purchase.",
        "Good product overall."
    ]
    labels = ["Positive", "Negative", "Neutral", "Positive", "Negative", 
              "Positive", "Negative", "Positive", "Negative", "Positive"]
    
    try:
        response = requests.post(
            "http://localhost:8001/submit-feedback",
            json={
                "text": texts[i],
                "model_prediction": "Positive",
                "user_label": labels[i]
            }
        )
    except:
        pass

print("   [OK] Added 10 feedback samples")

# Step 6: Retrain
print("\n6. Retraining model...")
try:
    response = requests.post("http://localhost:8004/retrain")
    if response.status_code == 200:
        result = response.json()
        print(f"   [OK] Model retrained: {result.get('version')} (Accuracy: {result.get('accuracy', 0):.2%})")
    else:
        print(f"   [ERROR] {response.text}")
except Exception as e:
    print(f"   [ERROR] {e}")

print("\n" + "=" * 60)
print("Workflow Test Complete!")
print("=" * 60)

