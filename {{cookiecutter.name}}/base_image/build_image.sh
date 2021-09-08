#!/bin/bash

set -euo pipefail

INPUT_DIR="$1"
IMAGE="$2"

OUTPUT_DIR="./dist"
if [[ ! -e "${OUTPUT_DIR}" ]] ; then
    mkdir -p "${OUTPUT_DIR}"
fi

cp -R "${INPUT_DIR}" "${OUTPUT_DIR}"
docker build -t "${IMAGE}" .
