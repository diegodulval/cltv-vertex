# Kubeflow Pipelines template repository

This folder contains an example structure for creating pipelines with CI/CD integration.

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

## Makefile commands
You can make use of the commands defined in the `Makefile` to perform different operations. The commands that you should type in the terminal are always of the following pattern:

```sh
# common pattern
make [specific command]
```
For instance, to initiate the project (i.e. to install the requiered dependencies and install pre-commit hooks), run the following:

```sh
# Install dependencies
make setup_env
```
