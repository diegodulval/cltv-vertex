import importlib
import logging

import click

from deploy.settings import PipelineSettings
from deploy.vertex_ai import run_pipeline, schedule_pipeline


def _setup_logging(level=logging.INFO, path=None):
    """Set up streamed logging and optionally file logging"""
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s|%(message)s")
    if path:
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def _load_pipeline(pipeline):
    """Load the specified pipeline function"""
    pipeline_module, pipeline_func = pipeline.rsplit(".", 1)
    module = importlib.import_module(pipeline_module)
    pipeline = getattr(module, pipeline_func)
    return pipeline


@click.group()
@click.pass_context
@click.version_option()
@click.option(
    "-v",
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Set the verbosity of logging",
)
@click.option(
    "-l",
    "--log-file",
    default=None,
    show_default=True,
    help="Set the log file location",
)
def cli(ctx, log_level, log_file):
    """Deploy - Helper script for running pipelines via Google Vertex AI"""
    _setup_logging(log_level, log_file)
    ctx.ensure_object(dict)
    ctx.obj["_logger"] = logging.getLogger(__name__)


@cli.command()
@click.pass_context
@click.option(
    "-s",
    "--settings",
    default=None,
    show_default=True,
    type=click.Path(exists=True),
    help="Pipeline settings file location",
)
@click.argument("pipeline")
def run(ctx, pipeline, settings):
    """Run the specified pipeline"""
    logger = ctx.obj["_logger"]
    logger.info("Running pipeline '{}'".format(pipeline))
    pipeline = _load_pipeline(pipeline)
    settings = PipelineSettings(pipeline, settings)
    run_pipeline(
        pipeline,
        project_id=settings.project_id,
        region=settings.region,
        pipeline_root=settings.pipeline_root,
        parameter_values=settings.pipeline_arguments,
        service_account=settings.service_account,
    )
    logger.info("Done")


@cli.command()
@click.pass_context
@click.option(
    "-s",
    "--settings",
    default=None,
    show_default=True,
    type=click.Path(exists=True),
    help="Pipeline settings file location",
)
@click.argument("pipeline")
def schedule(ctx, pipeline, settings):
    """Schedule the specified pipeline"""
    logger = ctx.obj["_logger"]
    logger.info("Scheduling pipeline '{}'".format(pipeline))
    pipeline = _load_pipeline(pipeline)
    settings = PipelineSettings(pipeline, settings)
    schedule_pipeline(
        pipeline,
        schedule=settings.run_schedule,
        project_id=settings.project_id,
        region=settings.region,
        pipeline_root=settings.pipeline_root,
        parameter_values=settings.pipeline_arguments,
        service_account=settings.service_account,
    )
    logger.info("Done")
