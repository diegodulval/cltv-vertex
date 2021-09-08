from datetime import datetime

from deploy.settings import PipelineSettings
from deploy.vertex_ai import run_pipeline

import kfp
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.v2.dsl import Dataset, Output, component

PROJECT_SETTINGS = PipelineSettings("pipelines.training.pipeline")
PROJECT_ID = PROJECT_SETTINGS.project_id
REGION = PROJECT_SETTINGS.region
PIPELINE_ROOT = PROJECT_SETTINGS.pipeline_root
BASE_IMAGE = PROJECT_SETTINGS.base_image
PROJECT_NAME = PROJECT_SETTINGS.project_name

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
MODEL_DISPLAY_NAME = f"{PROJECT_NAME}_train_deploy_{TIMESTAMP}"
SERVING_CONTAINER_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"


@component(base_image=BASE_IMAGE)
def example_component_op(data: Output[Dataset]):
    """This is an example on how to turn Python code into a Kubeflow component"""
    from datetime import datetime
    import logging

    logging.info(f"The current time is {datetime.now()}, the path of the output is {str(data.path)}")


@component(base_image=BASE_IMAGE)
def load_dataset_op(data: Output[Dataset]):
    from trainer.prepare_data import load_dataset

    load_dataset(data.path)


@kfp.dsl.pipeline(
    name="modelling-pipeline-" + TIMESTAMP,
    description="Training and deployment pipeline",
)
def pipeline(
    model_display_name: str = MODEL_DISPLAY_NAME,
    serving_container_image_uri: str = SERVING_CONTAINER_IMAGE,
    project: str = PROJECT_ID,
    pipeline_root: str = PIPELINE_ROOT,
):

    load_dataset_task = load_dataset_op()

    train_task = gcc_aip.CustomContainerTrainingJobRunOp(
        project=project,
        display_name=f"{PROJECT_NAME}-pipelines-created-job",
        container_uri=BASE_IMAGE,
        model_serving_container_image_uri=serving_container_image_uri,
        accelerator_type="NVIDIA_TESLA_P4",
        accelerator_count=1,
        machine_type="n1-standard-4",
        staging_bucket=pipeline_root,
        replica_count=1
    )
    train_task.after(load_dataset_task)

    endpoint_create_task = gcc_aip.EndpointCreateOp(
        project=project,
        display_name=f"{PROJECT_NAME}-pipelines-created-endpoint",
    )
    endpoint_create_task.after(train_task)

    model_deploy_task = gcc_aip.ModelDeployOp(  # noqa: F841
        project=project,
        endpoint=endpoint_create_task.outputs["endpoint"],
        model=train_task.outputs["model"],
        deployed_model_display_name=model_display_name,
        machine_type="n1-standard-2",
    )


if __name__ == "__main__":
    run_pipeline(pipeline, PROJECT_ID, REGION, PIPELINE_ROOT)
