.PHONY = all clean

ROOT_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
VENV_DIR = ${ROOT_DIR}/venv

GCR_HOST = "{{cookiecutter._internal_vars.region|gcr_location|lower}}.gcr.io"
PROJECT_ID = "{{cookiecutter._internal_vars.default_project}}"
TAG_NAME = "latest"
IMAGE_NAME = "${GCR_HOST}/${PROJECT_ID}/project-base:${TAG_NAME}"

all: setup_env test build_image push_image run

${VENV_DIR}:
	@echo "Setting up local environment..."
	virtualenv -p python3 "${VENV_DIR}"
	${VENV_DIR}/bin/pip install --upgrade pip setuptools
	${VENV_DIR}/bin/pip install -r "${ROOT_DIR}/base_image/requirements.txt"
	${VENV_DIR}/bin/pip install -r "${ROOT_DIR}/base_image/dev-requirements.txt"
	${VENV_DIR}/bin/pip install "${ROOT_DIR}/deploy"
	${VENV_DIR}/bin/pip install -e "${ROOT_DIR}/model_package"

setup_env: ${VENV_DIR}

test: setup_env
	@echo "Running tests..."
	${VENV_DIR}/bin/python -m pytest -v ${ROOT_DIR}/model_package

build_image: setup_env
	@echo "Building Docker base image..."
	cd ${ROOT_DIR}/model_package/ && ${VENV_DIR}/bin/python setup.py sdist
	cd ${ROOT_DIR}/base_image/ && ./build_image.sh "${ROOT_DIR}/model_package/dist" ${IMAGE_NAME}

push_image: build_image
	@echo "Pushing Docker base image..."
	docker push ${IMAGE_NAME}

run: push_image
	@echo "Deploying pipelines..."
	PYTHONPATH=${ROOT_DIR}/pipelines:${PYTHONPATH} ${VENV_DIR}/bin/deploy run pipelines.training.pipeline

clean:
	@echo "Cleaning up..."
	rm -rf "${VENV_DIR}"
	rm -rf "${ROOT_DIR}/model_package/dist"
	rm -rf "${ROOT_DIR}/base_image/dist"
