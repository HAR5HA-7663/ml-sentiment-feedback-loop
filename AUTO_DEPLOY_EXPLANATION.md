# Auto-Deploy Implementation Explanation

## Current Implementation (Fixed)

### How It Works:

1. **API Call**: `POST /bootstrap?auto_deploy=true`
2. **Training Job Created**: SageMaker training job starts
3. **Background Thread Started**: A daemon thread starts monitoring the training job
4. **Polling**: Thread checks training status every 60 seconds
5. **Auto-Deploy**: When status = "Completed", automatically deploys model to endpoint

### Code Flow:

```python
# In bootstrap endpoint:
if auto_deploy:
    monitor_and_deploy_sync(training_job_name)  # Starts thread immediately

# Thread function:
def monitor_and_deploy_sync(job_name):
    thread = threading.Thread(target=run_async, daemon=True)
    thread.start()  # Runs in background

# Async monitoring:
async def monitor_and_deploy(job_name):
    while elapsed < max_wait_time:
        status = check_training_status(job_name)
        if status == 'Completed':
            deploy_model(job_name)
            return
        await asyncio.sleep(60)
```

### Why Previous Version Failed:

- **FastAPI BackgroundTasks** runs tasks after response is sent
- In ECS containers, tasks can be lost if container restarts
- No persistence or retry mechanism
- Background task never started (no logs found)

### Current Fix:

- **Threading approach**: Starts thread immediately, more reliable
- **Daemon thread**: Doesn't block container shutdown
- **Better error handling**: Logs errors to CloudWatch
- **Still has limitations**: Thread dies if container restarts

---

## Better Long-Term Solution: EventBridge + Lambda

### Architecture:

```
SageMaker Training Job
    ↓ (completes)
EventBridge Rule (SageMaker Training State Change)
    ↓ (triggers)
Lambda Function
    ↓ (calls)
Model Init Service API /deploy endpoint
    ↓
Endpoint Updated
```

### Benefits:

- ✅ **Reliable**: EventBridge triggers are guaranteed
- ✅ **Persistent**: No dependency on container lifecycle
- ✅ **Scalable**: Works even if service restarts
- ✅ **AWS Native**: Uses AWS event-driven architecture

### Implementation Steps:

1. **Create EventBridge Rule** (in Terraform):

```hcl
resource "aws_cloudwatch_event_rule" "sagemaker_training_complete" {
  name        = "${var.project_name}-training-complete"
  description = "Trigger when SageMaker training job completes"

  event_pattern = jsonencode({
    source      = ["aws.sagemaker"]
    detail-type = ["SageMaker Training Job State Change"]
    detail = {
      TrainingJobStatus = ["Completed"]
      TrainingJobName = {
        prefix = "${var.project_name}-training-"
      }
    }
  })
}
```

2. **Create Lambda Function** (in Terraform):

```hcl
resource "aws_lambda_function" "auto_deploy" {
  filename      = "auto_deploy.zip"
  function_name = "${var.project_name}-auto-deploy"
  role          = aws_iam_role.lambda_auto_deploy.arn
  handler       = "index.handler"
  runtime       = "python3.11"

  environment {
    variables = {
      MODEL_INIT_SERVICE_URL = "http://model-init-service:8006"
      PROJECT_NAME           = var.project_name
    }
  }
}
```

3. **Lambda Code**:

```python
import json
import boto3
import requests

def handler(event, context):
    # Extract training job name from EventBridge event
    job_name = event['detail']['TrainingJobName']

    # Call model-init-service deploy endpoint
    service_url = os.environ['MODEL_INIT_SERVICE_URL']
    response = requests.post(
        f"{service_url}/deploy/{job_name}",
        timeout=30
    )

    return {
        'statusCode': 200,
        'body': json.dumps(f'Deployment triggered for {job_name}')
    }
```

4. **Connect EventBridge to Lambda**:

```hcl
resource "aws_cloudwatch_event_target" "lambda_target" {
  rule      = aws_cloudwatch_event_rule.sagemaker_training_complete.name
  target_id = "AutoDeployLambda"
  arn       = aws_lambda_function.auto_deploy.arn
}
```

---

## Current Status

✅ **Fixed**: Thread-based background monitoring (more reliable)
⚠️ **Limitation**: Still depends on container staying alive
🔮 **Future**: EventBridge + Lambda solution (production-ready)

---

## Testing Auto-Deploy

1. Start training with auto-deploy:

```bash
curl -X POST "http://ALB_URL/model-init/bootstrap?auto_deploy=true"
```

2. Check logs for background thread:

```bash
aws logs tail /ecs/ml-sentiment/model-init-service --filter-pattern "auto-deploy"
```

3. Wait for training (15-20 min), then check endpoint:

```bash
curl http://ALB_URL/model-init/endpoint-status
```

Expected: Endpoint `last_modified_time` should update automatically when training completes.
