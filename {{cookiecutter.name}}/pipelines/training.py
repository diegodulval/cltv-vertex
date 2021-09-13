
import os
import sys

sys.path.append("..") 
from model_package.src.trainer.prepare_data import load_dataset_vtx
from model_package.src.trainer.train import train_model_vtx



from datetime import datetime
from deploy.settings import PipelineSettings
from deploy.vertex_ai import run_pipeline
import kfp
from google_cloud_pipeline_components import aiplatform as gcc_aip



import kfp
import kfp.components as comp
import kfp.dsl as dsl

from kfp.v2 import compiler
from kfp.v2.google.client import AIPlatformClient

from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
)

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
    pipeline_root='gs://gme-e2e-mlops-featurestore-sandbox',
    )
def train_and_evaluate_vtx(
        config_path: str = 'gs://gme-e2e-mlops-featurestore-sandbox/config.yaml',
):
    
    loadOp = load_dataset_vtx(config_path=config_path)
    trainOp = train_model_vtx(config_path=config_path,
                              cleaned_data=loadOp.outputs['cleaned_data'])


if __name__ == "__main__":
    print("----")
    print(PROJECT_ID)
    print("-----")
    run_pipeline(train_and_evaluate_vtx, PROJECT_ID, REGION, PIPELINE_ROOT)
