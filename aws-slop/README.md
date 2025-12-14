# AWS Deployment Scripts and Documentation

This folder contains all deployment scripts, testing utilities, and AWS-specific documentation.

## Structure

- `scripts/` - PowerShell and Python scripts for deployment, testing, and maintenance
- `docs/` - AWS deployment guides and documentation

## Quick Reference

### Deployment Scripts

- `deploy-direct.ps1` - Fast direct deployment to ECS (2-3 min vs 20-30 min)
- `deploy-inference.ps1` - Deploy inference service specifically
- `convert-tokenizer-docker.ps1` - Convert tokenizer pickle to JSON using Docker

### Testing Scripts

- `test-full-flow.ps1` - Test complete ML feedback loop
- `check-ready.ps1` - Check endpoint status and test predictions
- `test-api.html` - Web-based API testing interface

### Training Scripts

- `trigger-training.ps1` - Start SageMaker training job
- `monitor-training.ps1` - Monitor training job progress
- `wait-and-train.ps1` - Automated training trigger and monitoring

### Documentation

- `SAGEMAKER_GUIDE.md` - Complete SageMaker integration guide
- `READY_TO_TRAIN.md` - Training setup instructions
- `TESTING.md` - Testing procedures and endpoints

## Usage

During development, use `deploy-direct.ps1` for fast iteration.
For production, use GitHub Actions CI/CD pipeline.
