# Application Load Balancer for ML Sentiment System

# Security Group for ALB
resource "aws_security_group" "alb" {
  name        = "${var.project_name}-alb-sg"
  description = "Security group for Application Load Balancer"
  vpc_id      = var.vpc_id

  # Allow inbound HTTP from anywhere
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow HTTP from anywhere"
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
    Name = "${var.project_name}-alb-sg"
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.subnet_ids

  enable_deletion_protection = false
  enable_http2              = true

  tags = {
    Name = "${var.project_name}-alb"
  }
}

# Target Groups (one per service)
resource "aws_lb_target_group" "services" {
  for_each = { for svc in var.services : svc.name => svc }

  name     = trim(substr("${var.project_name}-${each.value.name}-tg", 0, 32), "-")
  port     = each.value.port
  protocol = "HTTP"
  vpc_id   = var.vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    path                = each.value.health_path
    protocol            = "HTTP"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }

  deregistration_delay = 30

  tags = {
    Name = "${var.project_name}-${each.value.name}-tg"
  }
}

# ALB Listener (HTTP on port 80)
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  # Default action - forward to API gateway
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.services["api-gateway-service"].arn
  }
}

# Listener Rules for routing
resource "aws_lb_listener_rule" "predict" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.services["inference-service"].arn
  }

  condition {
    path_pattern {
      values = ["/predict*"]
    }
  }
}

resource "aws_lb_listener_rule" "feedback" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 101

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.services["feedback-service"].arn
  }

  condition {
    path_pattern {
      values = ["/feedback*", "/submit-feedback*"]
    }
  }
}

resource "aws_lb_listener_rule" "models" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 102

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.services["model-registry-service"].arn
  }

  condition {
    path_pattern {
      values = ["/models*", "/register-model*"]
    }
  }
}

resource "aws_lb_listener_rule" "evaluate" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 103

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.services["evaluation-service"].arn
  }

  condition {
    path_pattern {
      values = ["/evaluate*", "/run-evaluation*"]
    }
  }
}

resource "aws_lb_listener_rule" "retrain" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 104

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.services["retraining-service"].arn
  }

  condition {
    path_pattern {
      values = ["/retrain*"]
    }
  }
}
