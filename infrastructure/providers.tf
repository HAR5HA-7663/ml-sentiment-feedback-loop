provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "ML-Sentiment-Feedback-Loop"
      Environment = "demo"
      ManagedBy   = "Terraform"
      Owner       = "HAR5HA-7663"
    }
  }
}
