output "models_bucket_name" {
  description = "Name of the models S3 bucket"
  value       = aws_s3_bucket.models.id
}

output "models_bucket_arn" {
  description = "ARN of the models S3 bucket"
  value       = aws_s3_bucket.models.arn
}

output "data_bucket_name" {
  description = "Name of the data S3 bucket"
  value       = aws_s3_bucket.data.id
}

output "data_bucket_arn" {
  description = "ARN of the data S3 bucket"
  value       = aws_s3_bucket.data.arn
}

output "artifacts_bucket_name" {
  description = "Name of the artifacts S3 bucket"
  value       = aws_s3_bucket.artifacts.id
}

output "artifacts_bucket_arn" {
  description = "ARN of the artifacts S3 bucket"
  value       = aws_s3_bucket.artifacts.arn
}
