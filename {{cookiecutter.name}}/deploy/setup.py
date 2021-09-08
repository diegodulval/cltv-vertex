from setuptools import find_packages, setup

REQUIRES = [
    "click",
    "google-cloud-aiplatform",
    "google-cloud-pipeline-components",
    "kfp",
]

setup(
    name="deploy",
    version="0.0.1",
    description="Deployment helper script",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    install_requires=REQUIRES,
    entry_points={
        "console_scripts": [
            "deploy = deploy.__main__:cli",
        ],
    },
)
