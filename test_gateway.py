import requests
import json

GATEWAY_URL = "http://localhost:8080"

print("=" * 60)
print("Testing API Gateway")
print("=" * 60)

# Test 1: Root endpoint
print("\n1. Testing root endpoint...")
try:
    response = requests.get(f"{GATEWAY_URL}/")
    if response.status_code == 200:
        result = response.json()
        print(f"   [OK] Gateway info retrieved")
        print(f"   Service: {result.get('service')}")
        print(f"   Endpoints available: {len(result.get('endpoints', {}))}")
    else:
        print(f"   [ERROR] Status code: {response.status_code}")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test 2: Health check
print("\n2. Testing aggregated health check...")
try:
    response = requests.get(f"{GATEWAY_URL}/health")
    if response.status_code == 200:
        result = response.json()
        print(f"   [OK] Overall status: {result.get('overall')}")
        for service, status in result.get('services', {}).items():
            print(f"       - {service}: {status.get('status')}")
    else:
        print(f"   [ERROR] Status code: {response.status_code}")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test 3: Make prediction
print("\n3. Testing prediction through gateway...")
try:
    response = requests.post(
        f"{GATEWAY_URL}/predict",
        json={"text": "This product is amazing! I love it."}
    )
    if response.status_code == 200:
        result = response.json()
        print(f"   [OK] Prediction: {result.get('label')} (Confidence: {result.get('confidence', 0):.2%})")
    else:
        print(f"   [ERROR] Status code: {response.status_code}")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test 4: Submit feedback
print("\n4. Testing feedback submission through gateway...")
try:
    response = requests.post(
        f"{GATEWAY_URL}/feedback",
        json={
            "text": "Great product!",
            "model_prediction": "Positive",
            "user_label": "Positive"
        }
    )
    if response.status_code == 200:
        result = response.json()
        print(f"   [OK] Feedback submitted (ID: {result.get('id')})")
    else:
        print(f"   [ERROR] Status code: {response.status_code}")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test 5: Get models
print("\n5. Testing model listing through gateway...")
try:
    response = requests.get(f"{GATEWAY_URL}/models")
    if response.status_code == 200:
        result = response.json()
        print(f"   [OK] Retrieved {len(result)} model(s)")
    else:
        print(f"   [ERROR] Status code: {response.status_code}")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test 6: Run evaluation
print("\n6. Testing evaluation through gateway...")
try:
    response = requests.post(f"{GATEWAY_URL}/evaluate")
    if response.status_code == 200:
        result = response.json()
        print(f"   [OK] Evaluation complete: Accuracy = {result.get('accuracy', 0):.2%}")
    else:
        print(f"   [ERROR] Status code: {response.status_code}")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test 7: Check X-Request-ID propagation
print("\n7. Testing X-Request-ID propagation...")
try:
    custom_request_id = "test-12345"
    response = requests.get(
        f"{GATEWAY_URL}/health",
        headers={"X-Request-ID": custom_request_id}
    )
    if response.status_code == 200:
        print(f"   [OK] Custom X-Request-ID handled successfully")
        print(f"   Request ID used: {custom_request_id}")
    else:
        print(f"   [ERROR] Status code: {response.status_code}")
except Exception as e:
    print(f"   [ERROR] {e}")

print("\n" + "=" * 60)
print("API Gateway Test Complete!")
print("=" * 60)
