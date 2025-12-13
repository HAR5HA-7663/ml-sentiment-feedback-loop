import requests
import uuid

GATEWAY_URL = "http://localhost:8080"

print("=" * 80)
print("Testing Logging and Request Tracing")
print("=" * 80)

# Generate a custom request ID to trace
custom_request_id = str(uuid.uuid4())
print(f"\nUsing Request ID: {custom_request_id}")
print(f"Watch the Docker logs for this ID to see it flow through services")
print(f"\nCommand to trace: docker-compose logs | grep \"{custom_request_id}\"\n")
print("=" * 80)

headers = {
    "X-Request-ID": custom_request_id,
    "Content-Type": "application/json"
}

# Test 1: Make a prediction
print("\n1. Making prediction (will route through API Gateway → Inference Service)")
print("-" * 80)
try:
    response = requests.post(
        f"{GATEWAY_URL}/predict",
        headers=headers,
        json={"text": "This product is absolutely amazing! Love it!"}
    )
    
    if response.status_code == 200:
        result = response.json()
        returned_request_id = response.headers.get("X-Request-ID")
        print(f"✅ Success!")
        print(f"   Prediction: {result.get('label')}")
        print(f"   Confidence: {result.get('confidence', 0):.4f}")
        print(f"   Returned Request ID: {returned_request_id}")
        print(f"   Match: {'✅' if returned_request_id == custom_request_id else '❌'}")
    else:
        print(f"❌ Error: Status {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Submit feedback
print("\n2. Submitting feedback (will route through API Gateway → Feedback Service)")
print("-" * 80)
try:
    response = requests.post(
        f"{GATEWAY_URL}/feedback",
        headers=headers,
        json={
            "text": "Great product!",
            "model_prediction": "Positive",
            "user_label": "Positive"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        returned_request_id = response.headers.get("X-Request-ID")
        print(f"✅ Success!")
        print(f"   Feedback ID: {result.get('id')}")
        print(f"   Returned Request ID: {returned_request_id}")
        print(f"   Match: {'✅' if returned_request_id == custom_request_id else '❌'}")
    else:
        print(f"❌ Error: Status {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Run evaluation
print("\n3. Running evaluation (will route through API Gateway → Evaluation Service)")
print("-" * 80)
try:
    response = requests.post(
        f"{GATEWAY_URL}/evaluate",
        headers=headers
    )
    
    if response.status_code == 200:
        result = response.json()
        returned_request_id = response.headers.get("X-Request-ID")
        print(f"✅ Success!")
        print(f"   Accuracy: {result.get('accuracy', 0):.4f}")
        print(f"   Samples: {result.get('total', 0)}")
        print(f"   Returned Request ID: {returned_request_id}")
        print(f"   Match: {'✅' if returned_request_id == custom_request_id else '❌'}")
    else:
        print(f"❌ Error: Status {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
print("\nTo see the request trace in the logs, run:")
print(f'docker-compose logs | grep "{custom_request_id}"')
print("\nYou should see logs from:")
print("  - api-gateway (3 requests)")
print("  - inference-service (1 request)")
print("  - feedback-service (1 request)")
print("  - evaluation-service (1 request)")
print("\nAll with the same Request ID!")
print("=" * 80)
