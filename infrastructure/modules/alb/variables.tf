variable "project_name" {
  description = "Project name prefix"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for ALB"
  type        = list(string)
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
