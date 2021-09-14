import os
import sys
from datetime import datetime
import kfp.v2.dsl as dsl
from kfp.v2 import compiler

from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
    Model
)

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


def batch_prediction_vtx_component(base_image, aliz_aip_project,project,runner,gcs_source,bq_source_sql,bq_source_table,gcs_destination,bq_destination_table,model_uri,staging_location,temp_location,experiment):

    @component(base_image=base_image)
    def batch_prediction_vtx(
            project: str,
            runner: str,
            gcs_source: str,
            bq_source_sql: str,
            bq_source_table: str,
            gcs_destination: str,
            bq_destination_table: str,
            model_uri: str,
            staging_location: str,
            temp_location: str,
            experiment: str,
            aliz_aip_project: str,
    ):
        """Batch prediction callback function used specifically in Kubeflow pipeline.

        Arguments:
            project (str): Google Cloud Platform Projet ID
            runner (str): 'DirectRunner' or 'DataflowRunner'
            gcs_source (str): Google Cloud Storage source
            bq_source_sql (str): Google Big Query SQL source
            bq_source_table (str): Google Big Query fully qualified table name source
            gcs_destination (str): Google Cloud Storage destination
            bq_destination_table (str): Google Big Query fully qualified table name destination
            model_uri (str): Model URI
            staging_location (str): Google Cloud Storage staging storage location
            temp_location (str): Google Cloud Storage temporary storage location
            experiment (str): MLflow experiment name
        """
        import datetime
        import logging

        import apache_beam as beam
        import mlflow
        import numpy as np
        from apache_beam.options.pipeline_options import PipelineOptions

        class LoggerDoFn(beam.DoFn):
            def process(self, element, timestamp=beam.DoFn.TimestampParam, window=beam.DoFn.WindowParam, *args, **kwargs):
                logging.info("Element: %s, Timestamp: %s, Window: %s", str(element), str(timestamp), str(window))
                yield element

        class CustomPipelineOptions(PipelineOptions):
            @classmethod
            def _add_argparse_args(cls, parser):
                group_source = parser.add_mutually_exclusive_group()
                group_source.add_argument('--bq-source-table')
                group_source.add_argument('--bq-source-sql')
                group_source.add_argument('--gcs-source')

                group_destination = parser.add_mutually_exclusive_group()
                group_destination.add_argument('--bq-destination-table')
                group_destination.add_argument('--gcs-destination')

                parser.add_argument('--model-uri', required=True)

        class PredictDoFn(beam.DoFn):
            model = None

            def __init__(self, model_uri):
                self.model_uri = model_uri

            def setup(self):
                self.model = mlflow.pyfunc.load_model(self.model_uri)
                logging.info(f'loaded model {self.model}')

            def process(self, element, *args, **kwargs):
                logging.info(f'processing {element}')
                yield {
                    'timestamp': datetime.datetime.now(),
                    'model_uri': self.model_uri,
                    'original_row': element,
                    'prediction': self.model.predict(np.array(element.split(',')).astype(np.float).reshape(1, -1))[0].item()
                }

        def run():
            """
            assert runner in ['DirectRunner', 'DataflowRunner']

            assert (gcs_source != '' and bq_source_sql == '' and bq_source_table == '') or \
                (gcs_source == '' and bq_source_sql != '' and bq_source_table == '') or \
                (gcs_source == '' and bq_source_sql == '' and bq_source_table != '')

            assert (gcs_destination != '' and bq_destination_table == '') or \
                (gcs_destination == '' and bq_destination_table != '')
            """
            options = {
                'project': project,
                'staging_location': staging_location,
                'temp_location': temp_location,
                'model-uri': model_uri,
                'experiment': experiment,
            }

            if gcs_source != '':
                options['gcs-source'] = gcs_source
            elif bq_source_table != '':
                options['bq-source-table'] = bq_source_table
            elif bq_source_sql != '':
                options['bq-source-sql'] = bq_source_sql

            if gcs_destination != '':
                options['gcs-destination'] = gcs_destination
            elif bq_destination_table != '':
                options['bq-destination-table'] = bq_destination_table

            if runner == 'DirectRunner':
                pass
            else:
                repo = 'aip-batch-predictor'
                tag = 'latest'
                registry_host = 'eu.gcr.io'
                image_uri = f'{registry_host}/{project}/{repo}:{tag}'

                options.update({
                    'worker_harness_container_image': image_uri,
                    'region': 'europe-west3',
                })

            pipeline_options = CustomPipelineOptions.from_dictionary(options)
            with beam.Pipeline(runner=runner, options=pipeline_options) as p:

                # Load the data
                if pipeline_options.gcs_source:
                    raw_input = (
                                p | 'Read input from GCS' >> beam.io.ReadFromText(file_pattern=pipeline_options.gcs_source))
                elif pipeline_options.bq_source_table:
                    source_table = dict(zip(['project', 'dataset', 'table'], pipeline_options.bq_source_table.split('.')))
                    raw_input = (p | 'Read input from BQ table' >> beam.io.ReadFromBigQuery(**source_table))
                elif pipeline_options.bq_source_sql:
                    raw_input = (p | 'Read input from BQ SQL' >> beam.io.ReadFromBigQuery(
                        query=pipeline_options.bq_source_sql))

                # Do the prediction for each row and log it
                prediction = (raw_input
                            | 'Predict' >> beam.ParDo(PredictDoFn(pipeline_options.model_uri))
                            | "Log Predict" >> beam.ParDo(LoggerDoFn()))

                if pipeline_options.bq_destination_table:
                    dest_table = dict(
                        zip(['project', 'dataset', 'table'], pipeline_options.bq_destination_table.split('.')))
                    _ = prediction | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
                        **dest_table,
                        # SCHEMA_AUTODETECT here maybe or specify schema as a pipeline param as well?
                        schema='timestamp:TIMESTAMP,model_uri:STRING,original_row:STRING,prediction:FLOAT64',
                        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                        create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                        custom_gcs_temp_location='gs://dani_dataflow_sandbox_logdata_bucket'
                    )
                elif pipeline_options.gcs_destination:
                    # Generate "folder" in GCS for exact job run (uuid, timestamp?)
                    _ = (prediction
                        | 'Write to GCS' >> beam.io.WriteToText(file_path_prefix=pipeline_options.gcs_destination))

        logging.getLogger().setLevel(logging.INFO)
        run()

    return batch_prediction_vtx(
            project = project,
            runner = runner,
            gcs_source = gcs_source,
            bq_source_sql = bq_source_sql,
            bq_source_table = bq_source_table,
            gcs_destination = gcs_destination,
            bq_destination_table = bq_destination_table,
            model_uri = model_uri,
            staging_location = staging_location,
            temp_location = temp_location,
            experiment = experiment,
            aliz_aip_project = aliz_aip_project,
    )
    
if __name__ == "__main__":
    fire.Fire(batch_prediction_vtx_component)