import ast
import importlib
import os

import yaml


class PipelineSettings(object):
    """Class representing pipeline settings"""

    DEFAULT_SETTINGS_FILE = "pipeline_settings.yaml"
    PIPELINE_ENV_VAR_TEMPLATE = "PIPELINE_{}"

    @staticmethod
    def _get_pipeline_source(pipeline):
        """Get the source path of the specified pipeline (or assume provided input is already the source)"""
        if callable(pipeline):
            return "{}.{}".format(pipeline.__module__, pipeline.__name__)
        else:
            return str(pipeline)

    def __init__(self, pipeline, settings_path=None):
        self._source = PipelineSettings._get_pipeline_source(pipeline)
        self._source_path = importlib.import_module(
            self._source.rsplit(".", 1)[0]
        ).__file__
        if not settings_path:
            settings_path = os.path.join(
                os.path.dirname(self._source_path),
                PipelineSettings.DEFAULT_SETTINGS_FILE,
            )
        with open(settings_path, "r") as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)
        self._defaults = settings.get("defaults", dict())
        self._pipeline_settings = next(
            (
                i
                for i in settings.get("pipelines", list())
                if i["source"] == self._source
            ),
            dict(),
        )

    def __getattr__(self, name):
        """Get the value of a pipeline setting preference order: os env.var > specific setting > defaults."""
        env_var_name = PipelineSettings.PIPELINE_ENV_VAR_TEMPLATE.format(name.upper())
        if env_var_name in os.environ:
            value = os.environ[env_var_name]
            try:
                return ast.literal_eval(value)
            except Exception:
                return value
        if name in self._pipeline_settings:
            return self._pipeline_settings[name]
        if name in self._defaults:
            return self._defaults[name]
        raise Exception(
            "No setting '{}' configured for pipeline: '{}'".format(name, self._source)
        )
