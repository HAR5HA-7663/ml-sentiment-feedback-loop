import boto3
import os
import json

ecs = boto3.client('ecs')

CLUSTER_NAME = os.environ['ECS_CLUSTER_NAME']
SERVICE_NAMES = json.loads(os.environ['ECS_SERVICE_NAMES'])

def lambda_handler(event, context):
    """
    Auto-shutdown Lambda function
    Stops ECS services at night (11 PM) and starts them in the morning (7 AM)
    """
    action = event.get('action', 'stop')
    
    if action == 'stop':
        desired_count = 0
        print(f"Stopping ECS services in cluster: {CLUSTER_NAME}")
    else:
        desired_count = 1
        print(f"Starting ECS services in cluster: {CLUSTER_NAME}")
    
    results = []
    for service_name in SERVICE_NAMES:
        try:
            response = ecs.update_service(
                cluster=CLUSTER_NAME,
                service=service_name,
                desiredCount=desired_count
            )
            results.append({
                'service': service_name,
                'status': 'success',
                'desired_count': desired_count
            })
            print(f"✓ {service_name}: desired count set to {desired_count}")
        except Exception as e:
            results.append({
                'service': service_name,
                'status': 'error',
                'error': str(e)
            })
            print(f"✗ {service_name}: {str(e)}")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'action': action,
            'cluster': CLUSTER_NAME,
            'results': results
        })
    }
