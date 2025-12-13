# Lambda Function for Auto-Shutdown (11 PM - 7 AM)

# Archive Lambda function code
data "archive_file" "lambda_scheduler" {
  type        = "zip"
  source_file = "${path.module}/scheduler.py"
  output_path = "${path.module}/scheduler.zip"
}

# Lambda Function
resource "aws_lambda_function" "scheduler" {
  filename         = data.archive_file.lambda_scheduler.output_path
  function_name    = "${var.project_name}-ecs-scheduler"
  role             = var.lambda_role_arn
  handler          = "scheduler.lambda_handler"
  source_code_hash = data.archive_file.lambda_scheduler.output_base64sha256
  runtime          = "python3.11"
  timeout          = 60

  environment {
    variables = {
      ECS_CLUSTER_NAME  = var.ecs_cluster_name
      ECS_SERVICE_NAMES = jsonencode(var.ecs_service_names)
    }
  }

  tags = {
    Name = "${var.project_name}-ecs-scheduler"
  }
}

# EventBridge Rule - Stop services at 11 PM EST (4 AM UTC)
resource "aws_cloudwatch_event_rule" "stop_services" {
  name                = "${var.project_name}-stop-services"
  description         = "Stop ECS services at 11 PM EST"
  schedule_expression = "cron(0 4 * * ? *)"

  tags = {
    Name = "${var.project_name}-stop-services"
  }
}

resource "aws_cloudwatch_event_target" "stop_services" {
  rule      = aws_cloudwatch_event_rule.stop_services.name
  target_id = "StopECSServices"
  arn       = aws_lambda_function.scheduler.arn

  input = jsonencode({
    action = "stop"
  })
}

resource "aws_lambda_permission" "allow_stop_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridgeStop"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.stop_services.arn
}

# EventBridge Rule - Start services at 7 AM EST (12 PM UTC)
resource "aws_cloudwatch_event_rule" "start_services" {
  name                = "${var.project_name}-start-services"
  description         = "Start ECS services at 7 AM EST"
  schedule_expression = "cron(0 12 * * ? *)"

  tags = {
    Name = "${var.project_name}-start-services"
  }
}

resource "aws_cloudwatch_event_target" "start_services" {
  rule      = aws_cloudwatch_event_rule.start_services.name
  target_id = "StartECSServices"
  arn       = aws_lambda_function.scheduler.arn

  input = jsonencode({
    action = "start"
  })
}

resource "aws_lambda_permission" "allow_start_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridgeStart"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.start_services.arn
}

# CloudWatch Log Group for Lambda
resource "aws_cloudwatch_log_group" "lambda_scheduler" {
  name              = "/aws/lambda/${var.project_name}-ecs-scheduler"
  retention_in_days = 7

  tags = {
    Name = "${var.project_name}-lambda-scheduler-logs"
  }
}
