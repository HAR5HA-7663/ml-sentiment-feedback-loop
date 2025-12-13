output "alb_dns_name" {
  description = "Load Balancer DNS name - use this to access your API"
  value       = module.alb.alb_dns_name
}

output "alb_url" {
  description = "Full Load Balancer URL"
  value       = "http://${module.alb.alb_dns_name}"
}

output "s3_models_bucket" {
  description = "S3 bucket for storing trained models"
  value       = module.s3.models_bucket_name
}

output "s3_data_bucket" {
  description = "S3 bucket for training data and feedback"
  value       = module.s3.data_bucket_name
}

output "s3_artifacts_bucket" {
  description = "S3 bucket for artifacts and logs"
  value       = module.s3.artifacts_bucket_name
}

output "ecs_cluster_name" {
  description = "ECS Cluster name"
  value       = module.ecs.cluster_name
}

output "sagemaker_endpoint_name" {
  description = "SageMaker inference endpoint name"
  value       = "ml-sentiment-endpoint (will be created after first training)"
}

output "api_endpoints" {
  description = "API endpoints to test"
  value = {
    health    = "http://${module.alb.alb_dns_name}/health"
    predict   = "http://${module.alb.alb_dns_name}/predict"
    feedback  = "http://${module.alb.alb_dns_name}/feedback"
    models    = "http://${module.alb.alb_dns_name}/models"
    evaluate  = "http://${module.alb.alb_dns_name}/evaluate"
    retrain   = "http://${module.alb.alb_dns_name}/retrain"
  }
}
