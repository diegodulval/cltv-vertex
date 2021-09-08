import os
import tempfile
from datetime import datetime

from kfp.v2 import compiler
from kfp.v2.google.client import AIPlatformClient


def _get_pipeline_name(pipeline):
    """ Create unique pipeline name using a timestamp """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return "{}-{}".format(pipeline._component_human_name, timestamp)


def _get_internal_parameters(parameter_values, project_id, pipeline_root=None):
    """ Update user input parameters with internal values """
    parameter_values = parameter_values or dict()
    parameter_values["project"] = project_id
    if pipeline_root:
        parameter_values["pipeline_root"] = pipeline_root
    return parameter_values


def run_pipeline(pipeline, project_id, region, pipeline_root=None, parameter_values=None):
    """ Run the specified pipeline on Google Vertex AI """
    with tempfile.TemporaryDirectory() as d:
        output_file = os.path.join(d, "output.json")
        compiler.Compiler().compile(
            pipeline_func=pipeline,
            package_path=output_file,
            pipeline_name=_get_pipeline_name(pipeline),
        )
        api_client = AIPlatformClient(project_id=project_id, region=region)
        api_client.create_run_from_job_spec(
            job_spec_path=output_file,
            pipeline_root=pipeline_root,
            parameter_values=_get_internal_parameters(parameter_values, project_id, pipeline_root),
        )


def schedule_pipeline(pipeline, schedule, project_id, region, pipeline_root=None, parameter_values=None):
    """ Schedule the specified pipeline on Google Vertex AI """
    with tempfile.TemporaryDirectory() as d:
        output_file = os.path.join(d, "output.json")
        compiler.Compiler().compile(
            pipeline_func=pipeline,
            package_path=output_file,
        )
        api_client = AIPlatformClient(project_id=project_id, region=region)
        api_client.create_schedule_from_job_spec(
            job_spec_path=output_file,
            schedule=schedule,
            time_zone="Etc/UTC",
            pipeline_root=pipeline_root,
            parameter_values=_get_internal_parameters(parameter_values, project_id, pipeline_root),
        )
