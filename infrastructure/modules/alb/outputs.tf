output "alb_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}

output "alb_arn" {
  description = "ARN of the load balancer"
  value       = aws_lb.main.arn
}

output "alb_security_group_id" {
  description = "Security group ID of ALB"
  value       = aws_security_group.alb.id
}

output "target_group_arns" {
  description = "Map of service names to target group ARNs"
  value       = { for k, v in aws_lb_target_group.services : k => v.arn }
}
