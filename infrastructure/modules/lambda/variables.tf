variable "project_name" {
  description = "Project name prefix"
  type        = string
}

variable "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  type        = string
}

variable "ecs_service_names" {
  description = "List of ECS service names to manage"
  type        = list(string)
}

variable "lambda_role_arn" {
  description = "Lambda execution role ARN"
  type        = string
}
