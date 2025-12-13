output "lambda_function_name" {
  description = "Name of the Lambda scheduler function"
  value       = aws_lambda_function.scheduler.function_name
}

output "lambda_function_arn" {
  description = "ARN of the Lambda scheduler function"
  value       = aws_lambda_function.scheduler.arn
}
