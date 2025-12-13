variable "project_name" {
  description = "Project name prefix"
  type        = string
}

variable "account_id" {
  description = "AWS Account ID"
  type        = string
}

variable "models_bucket_arn" {
  description = "ARN of models S3 bucket"
  type        = string
}

variable "data_bucket_arn" {
  description = "ARN of data S3 bucket"
  type        = string
}

variable "artifacts_bucket_arn" {
  description = "ARN of artifacts S3 bucket"
  type        = string
}
