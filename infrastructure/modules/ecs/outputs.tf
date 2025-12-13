output "cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = aws_ecs_cluster.main.arn
}

output "service_names" {
  description = "List of ECS service names"
  value       = [for svc in aws_ecs_service.services : svc.name]
}

output "service_arns" {
  description = "Map of service names to ARNs"
  value       = { for k, v in aws_ecs_service.services : k => v.id }
}
