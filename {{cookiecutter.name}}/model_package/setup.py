from setuptools import setup, find_packages


setup(
    name="trainer",
    version="0.0.1",
    description="Project wide common code collection",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    install_requires=[],
    entry_points={},
)
