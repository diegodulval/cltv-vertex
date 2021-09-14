import os
import sys
from datetime import datetime
import kfp.v2.dsl as dsl
from kfp.v2 import compiler
from kfp.v2.google.client import AIPlatformClient

from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
    Model
)
from trainer.predict import batch_prediction_vtx_component
from deploy.settings import PipelineSettings
from deploy.vertex_ai import run_pipeline

# Defaults and environment settings
PROJECT_SETTINGS = PipelineSettings("pipelines.batch_prediction.batch_prediction_vtx")
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
    name='prediction',  # todo: CC
    description='Batch prediction pipeline'
    )
def batch_prediction_vtx(
        project: str = PROJECT_ID,
        pipeline_root: str = PIPELINE_ROOT,
        runner: str = 'DirectRunner',
        gcs_source: str = 'gs://dani_dataflow_sandbox_logdata_bucket/wine_data.csv',
        bq_source_sql: str = '',
        bq_source_table: str = '',
        gcs_destination: str = '',
        bq_destination_table: str = 'mlops-featurestore-sandbox.dani_dataflow_sandbox.predicted_output',
        model_uri: str = 'gs://dani-mlflow-test-bucket/1/ebe2652616b245599a9c436a02a2dee9/artifacts/model',
        staging_location: str = 'gs://dani_dataflow_sandbox_logdata_bucket/staging',
        temp_location: str = 'gs://dani_dataflow_sandbox_logdata_bucket/temp',
        experiment: str = 'experiment',
):
    batch_prediction_op = batch_prediction_vtx_component(
        base_image=BASE_IMAGE,
        aliz_aip_project=ALIZ_AIP_PROJECT,
        project=project,
        runner=runner,
        gcs_source=gcs_source,
        bq_source_table=bq_source_table,
        bq_source_sql=bq_source_sql,
        gcs_destination=gcs_destination,
        bq_destination_table=bq_destination_table,
        model_uri=model_uri,
        staging_location=staging_location,
        temp_location=temp_location,
        experiment=experiment,
    )
    

if __name__ == '__main__':
    print("----")
    print("\n"*10)
    print(PROJECT_ID)
    print(REGION)
    print(PIPELINE_ROOT)
    print("-----")

    run_pipeline(batch_prediction_vtx, PROJECT_ID, REGION, PIPELINE_ROOT)
