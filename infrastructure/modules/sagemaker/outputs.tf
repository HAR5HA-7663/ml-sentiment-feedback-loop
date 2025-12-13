output "endpoint_name" {
  description = "Name of the SageMaker endpoint"
  value       = aws_sagemaker_endpoint.sentiment.name
}

output "endpoint_arn" {
  description = "ARN of the SageMaker endpoint"
  value       = aws_sagemaker_endpoint.sentiment.arn
}

output "model_name" {
  description = "Name of the SageMaker model"
  value       = aws_sagemaker_model.sentiment.name
}
