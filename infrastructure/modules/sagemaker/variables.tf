variable "project_name" {
  description = "Project name prefix"
  type        = string
}

variable "sagemaker_role_arn" {
  description = "SageMaker execution role ARN"
  type        = string
}

variable "models_bucket" {
  description = "S3 bucket for models"
  type        = string
}

variable "data_bucket" {
  description = "S3 bucket for training data"
  type        = string
}
