# ECS Cluster and Services for ML Sentiment System

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "${var.project_name}-cluster"
  }
}

# Service Discovery Namespace (AWS Cloud Map)
resource "aws_service_discovery_private_dns_namespace" "main" {
  name        = "ml-sentiment.local"
  description = "Private DNS namespace for ML Sentiment services"
  vpc         = var.vpc_id

  tags = {
    Name = "${var.project_name}-service-discovery"
  }
}

# Service Discovery Service for each ECS service
resource "aws_service_discovery_service" "services" {
  for_each = { for svc in var.services : svc.name => svc }

  name = each.value.name

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id

    dns_records {
      ttl  = 10
      type = "A"
    }

    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }

  tags = {
    Name = "${var.project_name}-${each.value.name}-discovery"
  }
}

# CloudWatch Log Groups for each service
resource "aws_cloudwatch_log_group" "services" {
  for_each = { for svc in var.services : svc.name => svc }

  name              = "/ecs/${var.project_name}/${each.value.name}"
  retention_in_days = 7

  tags = {
    Name = "${var.project_name}-${each.value.name}-logs"
  }
}

# Security Group for ECS Tasks
resource "aws_security_group" "ecs_tasks" {
  name        = "${var.project_name}-ecs-tasks-sg"
  description = "Security group for ECS tasks"
  vpc_id      = var.vpc_id

  # Allow inbound from ALB
  ingress {
    from_port       = 0
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [var.alb_security_group_id]
    description     = "Allow traffic from ALB"
  }

  # Allow inbound within security group (service-to-service communication)
  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    self        = true
    description = "Allow traffic within ECS tasks"
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  tags = {
    Name = "${var.project_name}-ecs-tasks-sg"
  }
}

# ECS Task Definitions
resource "aws_ecs_task_definition" "services" {
  for_each = { for svc in var.services : svc.name => svc }

  family                   = "${var.project_name}-${each.value.name}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = var.execution_role_arn
  task_role_arn            = var.task_role_arn

  container_definitions = jsonencode([{
    name  = each.value.name
    image = "${var.ecr_registry}/ml-sentiment-${each.value.name}:${var.image_tag}"
    
    portMappings = [{
      containerPort = each.value.port
      protocol      = "tcp"
    }]

    environment = [
      {
        name  = "SERVICE_NAME"
        value = each.value.name
      },
      {
        name  = "SERVICE_PORT"
        value = tostring(each.value.port)
      },
      {
        name  = "AWS_REGION"
        value = data.aws_region.current.name
      },
      {
        name  = "S3_MODELS_BUCKET"
        value = var.models_bucket
      },
      {
        name  = "S3_DATA_BUCKET"
        value = var.data_bucket
      },
      {
        name  = "S3_ARTIFACTS_BUCKET"
        value = var.artifacts_bucket
      },
      {
        name  = "SAGEMAKER_ENDPOINT_NAME"
        value = var.sagemaker_endpoint
      },
      {
        name  = "SAGEMAKER_ROLE_ARN"
        value = var.sagemaker_role_arn
      },
      {
        name  = "PROJECT_NAME"
        value = var.project_name
      },
      # Service discovery URLs
      {
        name  = "INFERENCE_SERVICE_URL"
        value = "http://inference-service.ml-sentiment.local:8000"
      },
      {
        name  = "FEEDBACK_SERVICE_URL"
        value = "http://feedback-service.ml-sentiment.local:8001"
      },
      {
        name  = "MODEL_REGISTRY_SERVICE_URL"
        value = "http://model-registry-service.ml-sentiment.local:8002"
      },
      {
        name  = "EVALUATION_SERVICE_URL"
        value = "http://evaluation-service.ml-sentiment.local:8003"
      },
      {
        name  = "RETRAINING_SERVICE_URL"
        value = "http://retraining-service.ml-sentiment.local:8004"
      },
      {
        name  = "NOTIFICATION_SERVICE_URL"
        value = "http://notification-service.ml-sentiment.local:8005"
      },
      {
        name  = "MODEL_INIT_SERVICE_URL"
        value = "http://model-init-service.ml-sentiment.local:8006"
      }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/${var.project_name}/${each.value.name}"
        "awslogs-region"        = data.aws_region.current.name
        "awslogs-stream-prefix" = "ecs"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:${each.value.port}${each.value.health_path} || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])

  tags = {
    Name = "${var.project_name}-${each.value.name}-task"
  }
}

# ECS Services
resource "aws_ecs_service" "services" {
  for_each = { for svc in var.services : svc.name => svc }

  name            = "${var.project_name}-${each.value.name}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.services[each.key].arn
  desired_count   = each.value.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = true
  }

  # Register with service discovery
  service_registries {
    registry_arn = aws_service_discovery_service.services[each.key].arn
  }

  # Only attach load balancer if service is exposed to ALB
  dynamic "load_balancer" {
    for_each = each.value.expose_to_alb ? [1] : []
    content {
      target_group_arn = var.alb_target_group_arns[each.key]
      container_name   = each.value.name
      container_port   = each.value.port
    }
  }

  deployment_maximum_percent         = 200
  deployment_minimum_healthy_percent = 50

  # Wait for ALB to be ready (only if exposed)
  depends_on = [var.alb_target_group_arns]

  tags = {
    Name = "${var.project_name}-${each.value.name}"
  }
}

# Data source for current AWS region
data "aws_region" "current" {}
