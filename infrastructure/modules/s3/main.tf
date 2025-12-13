# S3 Buckets for ML Sentiment System

# Models bucket - stores trained models
resource "aws_s3_bucket" "models" {
  bucket = "${var.project_name}-models-${var.account_id}"

  tags = {
    Name        = "${var.project_name}-models"
    Purpose     = "Store trained sentiment models"
    ContentType = "ML Models"
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Data bucket - stores training data and feedback
resource "aws_s3_bucket" "data" {
  bucket = "${var.project_name}-data-${var.account_id}"

  tags = {
    Name        = "${var.project_name}-data"
    Purpose     = "Store training data and user feedback"
    ContentType = "Training Data"
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Artifacts bucket - stores logs, metrics, reports
resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.project_name}-artifacts-${var.account_id}"

  tags = {
    Name        = "${var.project_name}-artifacts"
    Purpose     = "Store training artifacts, logs, and evaluation reports"
    ContentType = "Artifacts"
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Lifecycle policy to delete old artifacts after 30 days
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "delete-old-artifacts"
    status = "Enabled"

    filter {
      prefix = ""
    }

    expiration {
      days = 30
    }

    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }
}
