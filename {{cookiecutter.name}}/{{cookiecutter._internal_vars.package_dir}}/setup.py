from setuptools import find_packages, setup

setup(
    name="{{cookiecutter._internal_vars.package_name}}",
    version="0.0.1",
    description="Project wide common code collection",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    install_requires=[],
    entry_points={},
)
