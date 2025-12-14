variable "project_name" {
  description = "Project name prefix"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for ECS tasks"
  type        = list(string)
}

variable "ecr_registry" {
  description = "ECR registry URL"
  type        = string
}

variable "image_tag" {
  description = "Docker image tag"
  type        = string
}

variable "services" {
  description = "List of services"
  type = list(object({
    name          = string
    port          = number
    health_path   = string
    desired_count = number
    expose_to_alb = bool
  }))
}

variable "task_cpu" {
  description = "CPU units for tasks"
  type        = string
}

variable "task_memory" {
  description = "Memory for tasks"
  type        = string
}

variable "execution_role_arn" {
  description = "ECS execution role ARN"
  type        = string
}

variable "task_role_arn" {
  description = "ECS task role ARN"
  type        = string
}

variable "alb_target_group_arns" {
  description = "Map of service names to target group ARNs"
  type        = map(string)
}

variable "alb_security_group_id" {
  description = "Security group ID of ALB"
  type        = string
}

variable "models_bucket" {
  description = "S3 bucket for models"
  type        = string
}

variable "data_bucket" {
  description = "S3 bucket for data"
  type        = string
}

variable "artifacts_bucket" {
  description = "S3 bucket for artifacts"
  type        = string
}

variable "sagemaker_endpoint" {
  description = "SageMaker endpoint name"
  type        = string
}

variable "sagemaker_role_arn" {
  description = "SageMaker execution role ARN"
  type        = string
}
