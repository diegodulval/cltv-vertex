# Repository template

This directory contains an example repository structure that can be used as a template when building pipelines with Kubeflow.

It contains the pipeline definitions, written as Python scripts, and settings, written as two yaml files.

## Pipeline definitions
Here, we have two example pipeline definitions: a model training pipeline defined in `training.py` and a batch prediction pipeline defined in `batch_prediction.py`. 

The pipeline definition is a python function decorated with a `@dsl.pipeline` decorator. It consists of one or more components. For example, the `train_and_evaluate_vtx` pipeline is composed of `load_dataset_vtx_component` and `train_model_vtx_component`. These components are defined in files within the `model_package/src/trainer` directory.
 
## Pipeline settings
The file `pipeline_settings.yaml` contains the configurations of the pipeline. 

It contains two sections, the `default` section and the `pipelines` section.

The settings written in the `default` section will apply to **all** pipelines, unless overwritten by settings in the `pipelines` section.

In the `pipelines` section, multiple subsections, each corresponding to one specific pipeline, can be created. In our example, we have separate subsections for the training and the prediction pipelines, because each of them uses different base image for the components.

## Data-related configurations

This repository also contains `config.yaml`, which details all the data-related configurations for the modelling step.
These include settings about the target, data and context variables, the type of modeling to be performed ('regression' or 'classification'), the hyperparameters for the CatBoost model as well as the list of features to be used in the fitting and their data types. Do note that the list of features have to match exactly with the list of columns present in the source bigquery table, or the pipeline will fail. 


After setting the configurations, `config.yaml` should be uploaded to the same `pipeline_root` bucket defined in the `pipeline_settings.yaml`.
