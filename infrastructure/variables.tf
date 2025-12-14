variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-2"
}

variable "aws_account_id" {
  description = "AWS Account ID"
  type        = string
  default     = "143519759870"
}

variable "project_name" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "ml-sentiment"
}

variable "ecr_registry" {
  description = "ECR registry URL"
  type        = string
  default     = "143519759870.dkr.ecr.us-east-2.amazonaws.com"
}

variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

variable "vpc_id" {
  description = "VPC ID for ECS and ALB"
  type        = string
  default     = "vpc-0d7e9370b75899791"
}

variable "subnet_ids" {
  description = "Subnet IDs for ECS services and ALB"
  type        = list(string)
  default = [
    "subnet-0e12d8f8871222d19", # us-east-2a
    "subnet-0159f2bd6cb787bc4", # us-east-2b
    "subnet-0b5eb46b21dd80893"  # us-east-2c
  ]
}

variable "ecs_task_cpu" {
  description = "CPU units for ECS tasks (256 = 0.25 vCPU)"
  type        = string
  default     = "256"
}

variable "ecs_task_memory" {
  description = "Memory for ECS tasks (512 MB)"
  type        = string
  default     = "512"
}

variable "enable_auto_shutdown" {
  description = "Enable auto-shutdown Lambda (11 PM - 7 AM)"
  type        = bool
  default     = true
}

variable "services" {
  description = "List of microservices to deploy"
  type = list(object({
    name          = string
    port          = number
    health_path   = string
    desired_count = number
    expose_to_alb = bool  # Whether this service should be exposed via ALB
  }))
  default = [
    {
      name          = "api-gateway-service"
      port          = 8080
      health_path   = "/health"
      desired_count = 1
      expose_to_alb = true
    },
    {
      name          = "inference-service"
      port          = 8000
      health_path   = "/health"
      desired_count = 1
      expose_to_alb = false  # Internal only
    },
    {
      name          = "feedback-service"
      port          = 8001
      health_path   = "/health"
      desired_count = 1
      expose_to_alb = false  # Internal only
    },
    {
      name          = "model-registry-service"
      port          = 8002
      health_path   = "/health"
      desired_count = 1
      expose_to_alb = false  # Internal only
    },
    {
      name          = "evaluation-service"
      port          = 8003
      health_path   = "/health"
      desired_count = 1
      expose_to_alb = false  # Internal only
    },
    {
      name          = "retraining-service"
      port          = 8004
      health_path   = "/health"
      desired_count = 1
      expose_to_alb = false  # Internal only
    },
    {
      name          = "notification-service"
      port          = 8005
      health_path   = "/health"
      desired_count = 1
      expose_to_alb = false  # Internal service only
    },
    {
      name          = "model-init-service"
      port          = 8006
      health_path   = "/health"
      desired_count = 1
      expose_to_alb = false  # Internal service only
    }
  ]
}
