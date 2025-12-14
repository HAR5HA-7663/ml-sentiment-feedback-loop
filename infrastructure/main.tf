# Main Terraform configuration for ML Sentiment Feedback Loop

# S3 Buckets for models, data, and artifacts
module "s3" {
  source       = "./modules/s3"
  project_name = var.project_name
  account_id   = var.aws_account_id
}

# IAM Roles for ECS tasks, SageMaker, and Lambda
module "iam" {
  source              = "./modules/iam"
  project_name        = var.project_name
  account_id          = var.aws_account_id
  models_bucket_arn   = module.s3.models_bucket_arn
  data_bucket_arn     = module.s3.data_bucket_arn
  artifacts_bucket_arn = module.s3.artifacts_bucket_arn
}

# Application Load Balancer
module "alb" {
  source       = "./modules/alb"
  project_name = var.project_name
  vpc_id       = var.vpc_id
  subnet_ids   = var.subnet_ids
  services     = var.services
}

# ECS Cluster and Services
module "ecs" {
  source            = "./modules/ecs"
  project_name      = var.project_name
  vpc_id            = var.vpc_id
  subnet_ids        = var.subnet_ids
  ecr_registry      = var.ecr_registry
  image_tag         = var.image_tag
  services          = var.services
  task_cpu          = var.ecs_task_cpu
  task_memory       = var.ecs_task_memory
  execution_role_arn = module.iam.ecs_execution_role_arn
  task_role_arn     = module.iam.ecs_task_role_arn
  alb_target_group_arns = module.alb.target_group_arns
  alb_security_group_id = module.alb.alb_security_group_id
  
  # Environment variables for services
  models_bucket      = module.s3.models_bucket_name
  data_bucket        = module.s3.data_bucket_name
  artifacts_bucket   = module.s3.artifacts_bucket_name
  sagemaker_endpoint = "${var.project_name}-endpoint"
  sagemaker_role_arn = module.iam.sagemaker_role_arn
}

# SageMaker Training Pipeline and Endpoint
# Note: Model will be created dynamically by model-init-service
# Disabling Terraform-managed SageMaker to avoid chicken-egg problem
# module "sagemaker" {
#   source             = "./modules/sagemaker"
#   project_name       = var.project_name
#   sagemaker_role_arn = module.iam.sagemaker_role_arn
#   models_bucket      = module.s3.models_bucket_name
#   data_bucket        = module.s3.data_bucket_name
# }

# Auto-Shutdown Lambda (11 PM - 7 AM)
module "lambda" {
  count               = var.enable_auto_shutdown ? 1 : 0
  source              = "./modules/lambda"
  project_name        = var.project_name
  ecs_cluster_name    = module.ecs.cluster_name
  ecs_service_names   = module.ecs.service_names
  lambda_role_arn     = module.iam.lambda_role_arn
}
