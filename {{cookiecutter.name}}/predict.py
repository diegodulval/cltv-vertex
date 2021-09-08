import os
from google.cloud import aiplatform

from sklearn import datasets

digits = datasets.load_digits()
data = datasets.load_digits().data[0:100].reshape((-1, 64)).tolist()


REGION = "{{cookiecutter._internal_vars.region}}"
PROJECT_ID = os.getenv("PROJECT_ID", "{{cookiecutter._internal_vars.default_project}}")
ENDPOINT_NAME = "pipelines-created-endpoint"

aiplatform.init(project=PROJECT_ID, location=REGION)


ep = aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_NAME}"')
endpoint_name = ep[0].resource_name

end = aiplatform.Endpoint(
    endpoint_name=endpoint_name)

end.predict(data)
