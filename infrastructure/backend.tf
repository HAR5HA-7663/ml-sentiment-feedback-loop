terraform {
  backend "s3" {
    bucket         = "ml-sentiment-terraform-state-143519759870"
    key            = "ml-sentiment/terraform.tfstate"
    region         = "us-east-2"
    dynamodb_table = "ml-sentiment-terraform-locks"
    encrypt        = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  required_version = ">= 1.5.0"
}
