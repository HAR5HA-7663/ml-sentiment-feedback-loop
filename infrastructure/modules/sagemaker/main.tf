# SageMaker Training and Endpoint Configuration

# SageMaker Model (placeholder - will be replaced after training)
resource "aws_sagemaker_model" "sentiment" {
  name               = "${var.project_name}-sentiment-model"
  execution_role_arn = var.sagemaker_role_arn

  primary_container {
    image          = "763104351884.dkr.ecr.${data.aws_region.current.name}.amazonaws.com/tensorflow-inference:2.11-cpu"
    model_data_url = "s3://${var.models_bucket}/initial-model/model.tar.gz"
  }

  tags = {
    Name = "${var.project_name}-sentiment-model"
  }

  # This model will be replaced after first training
  lifecycle {
    ignore_changes = [primary_container]
  }
}

# SageMaker Endpoint Configuration
resource "aws_sagemaker_endpoint_configuration" "sentiment" {
  name = "${var.project_name}-endpoint-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.sentiment.name
    initial_instance_count = 1
    instance_type          = "ml.t2.medium"
    initial_variant_weight = 1.0
  }

  tags = {
    Name = "${var.project_name}-endpoint-config"
  }
}

# SageMaker Endpoint
resource "aws_sagemaker_endpoint" "sentiment" {
  name                 = "${var.project_name}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.sentiment.name

  tags = {
    Name = "${var.project_name}-endpoint"
  }
}

# Data source for current region
data "aws_region" "current" {}
