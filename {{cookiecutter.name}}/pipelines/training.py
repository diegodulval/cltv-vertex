import sys
from datetime import datetime

import kfp.v2.dsl as dsl
from deploy.settings import PipelineSettings
from deploy.vertex_ai import run_pipeline
from trainer.prepare_data import load_dataset_vtx_component
from trainer.train import train_model_vtx_component


PROJECT_SETTINGS = PipelineSettings("pipelines.training.train_and_evaluate_vtx")
PROJECT_ID = PROJECT_SETTINGS.project_id
REGION = PROJECT_SETTINGS.region
PIPELINE_ROOT = PROJECT_SETTINGS.pipeline_root
BASE_IMAGE = PROJECT_SETTINGS.base_image
PROJECT_NAME = PROJECT_SETTINGS.project_name
ALIZ_AIP_PROJECT = PROJECT_SETTINGS.aliz_aip_project

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
MODEL_DISPLAY_NAME = f"{PROJECT_NAME}_train_deploy_{TIMESTAMP}"
SERVING_CONTAINER_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"


@dsl.pipeline(
    name='train',
    description='Test training',
)
def train_and_evaluate_vtx(
    project: str = PROJECT_ID,
    pipeline_root: str = PIPELINE_ROOT,
    config_path: str = f'{PIPELINE_ROOT}/config.yaml',
):

    loadOp = load_dataset_vtx_component(config_path=config_path,
                                        base_image=BASE_IMAGE,
                                        aliz_aip_project=ALIZ_AIP_PROJECT)
    trainOp = train_model_vtx_component(config_path=config_path,
                                        cleaned_data=loadOp.outputs['cleaned_data'],
                                        base_image=BASE_IMAGE,
                                        aliz_aip_project=ALIZ_AIP_PROJECT)


if __name__ == "__main__":
    print("----")
    print(PROJECT_ID)
    print("-----")
    run_pipeline(train_and_evaluate_vtx, PROJECT_ID, REGION, PIPELINE_ROOT)
