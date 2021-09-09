#!/bin/bash

set -euo pipefail

PARGS_NAMES=("ALIZ_AIP_VERSION" "MLFLOW_IAP_VERSION" "OUTPUT_DIR")

function print_help() {
    cat <<EOF
Usage: $(basename ${BASH_SOURCE[0]}) ${PARGS_NAMES[@]}
EOF
}

if [[ -z "${PIP_CMD:-}" ]] ; then
    if [[ -n "$(which pip)" ]] && [[ -n "$(pip --version | grep "python 3")" ]] ; then
        PIP_CMD="pip"
    elif [[ -n "$(which pip3)" ]] ; then
        PIP_CMD="pip3"
    else
        add-apt-repository -y ppa:deadsnakes/ppa
        apt-get update
        apt-get install -y python3.6 python3-pip
        python3.6 -m pip install -U pip
        PIP_CMD="pip"
    fi
fi


function download_pkg() {
    $PIP_CMD download --extra-index-url "https://europe-python.pkg.dev/aliz-aip-dev/aip-python/simple/" --no-deps "$1"
}

if [[ -z "$(which gsutil)" ]] ; then
    echo "Error: gsutil not found"
    exit 1
fi

for p in "${PARGS_NAMES[@]}" ; do
    if [[ -z "${1:-}" ]] ; then
        print_help
        exit 1
    fi

    eval "$p='${1}'"
    shift
done

if [[ ! -e "${OUTPUT_DIR}" ]] ; then
    mkdir -p "${OUTPUT_DIR}"
fi

cd "${OUTPUT_DIR}"

$PIP_CMD install keyrings.google-artifactregistry-auth

download_pkg "mlflow-iap-plugin==${MLFLOW_IAP_VERSION}"
download_pkg "aliz-aip==${ALIZ_AIP_VERSION}"
