.PHONY = all clean

#ROOT_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
ROOT_DIR =  $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
VENV_DIR = ${ROOT_DIR}/venv

GCR_HOST = "gcr.io"
PROJECT_ID = "mlops-featurestore-sandbox"
TAG_NAME = "latest"
IMAGE_NAME = "${GCR_HOST}/${PROJECT_ID}/project-base:${TAG_NAME}"

all: setup_env test build_image push_image run

${VENV_DIR}:
	@echo "Setting up local environment..."
	virtualenv -p python3 "${VENV_DIR}"
	${VENV_DIR}/bin/pip install --upgrade pip setuptools
	${VENV_DIR}/bin/pip install -r "${ROOT_DIR}/base_image/requirements_train.txt"
	${VENV_DIR}/bin/pip install -r "${ROOT_DIR}/base_image/requirements_pred.txt"
	${VENV_DIR}/bin/pip install -r "${ROOT_DIR}/base_image/dev-requirements.txt"
	${VENV_DIR}/bin/pip install "${ROOT_DIR}/deploy"
	${VENV_DIR}/bin/pip install -e "${ROOT_DIR}/model_package"
	PIP_CMD="${VENV_DIR}/bin/pip" "${ROOT_DIR}/base_image/get_python_packages.sh" 1.1.8.var1 1.0.1 "${ROOT_DIR}/dist"
	${VENV_DIR}/bin/pip install "${ROOT_DIR}/dist/"*
	rm -rf "${ROOT_DIR}/dist/"
	${VENV_DIR}/bin/pre-commit install -t pre-commit
	${VENV_DIR}/bin/pre-commit install -t pre-push

setup_env: ${VENV_DIR}

test: setup_env
	@echo "Running tests..."
	${VENV_DIR}/bin/python -m pytest -v ${ROOT_DIR}/model_package

build_image: setup_env
	@echo "Building Docker base image..."
	PIP_CMD="${VENV_DIR}/bin/pip" "${ROOT_DIR}/base_image/get_python_packages.sh" 1.1.8.var1 1.0.1 "${ROOT_DIR}/dist"
	cd ${ROOT_DIR}/model_package/ && ${VENV_DIR}/bin/python setup.py sdist -d "${ROOT_DIR}/dist/"
	cd ${ROOT_DIR}/base_image/ && ./build_image.sh "${ROOT_DIR}/dist/" ${IMAGE_NAME}
	rm -rf "${ROOT_DIR}/dist/"

push_image: build_image
	@echo "Pushing Docker training base image..."
	docker push ${IMAGE_NAME}_train
	@echo "Pushing Docker prediction base image..."
	docker push ${IMAGE_NAME}_pred

run: push_image

run2:
	@echo "Deploying pipelines..."
	PYTHONPATH=${ROOT_DIR}/pipelines:${PYTHONPATH} ${VENV_DIR}/bin/deploy run pipelines.training.train_and_evaluate_vtx




clean:
	@echo "Cleaning up..."
	rm -rf "${VENV_DIR}"
	rm -rf "${ROOT_DIR}/model_package/dist"
	rm -rf "${ROOT_DIR}/base_image/dist"
