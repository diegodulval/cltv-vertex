# Kubeflow Pipelines template repository

This folder contains an example structure for creating pipelines with CI/CD integration.

## Setup
```sh
# Install dependencies
make setup_env

# Setup pre-commit and pre-push hooks
pre-commit install -t pre-commit
pre-commit install -t pre-push
```

## Project Structure

```
├── base_image                          <- A common container image that is used across all components in the pipeline.
├── deploy                              <- Deployment script for Google Vertex AI
├── model_package                       <- Project wide common code collection
│   ├── src                             <- Source code directory
│   └── test                            <- Unit tests code directory
│
├── pipelines
│   ├── pipeline_settings.yaml          <- Settings for each individual pipeline.
│   └── training.py                     <- Pipeline definition itself.
│
├── terraform                           <- Infrastructure component definition.
└── Makefile                            <- Development lifecycle management script
```
