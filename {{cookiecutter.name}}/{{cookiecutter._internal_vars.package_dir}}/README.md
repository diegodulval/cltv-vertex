# Project wide common code collection

## Summary

This is a Python package that contains project wide common code and is set up to be built with the project's base
image. This way code placed here will be available throughout the project regardless of the execution environment.

## Usage

Once the package is installed in a Python virtual environment or built into the base image, functions and objects
can be directly imported for use.

To install in a Python virtual environment (meant for testing and development):

1. Set up your environment
```shell
virtualenv -p python3 venv
source venv/bin/activate
pip install .
```

To use code from the collection (once installed in a virtual environment or built into the base image):

1. Import functions and object directly
```python
from {{cookiecutter._internal_vars.package_name}}.example import format_message
```
