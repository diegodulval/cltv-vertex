#!/bin/bash

set -euo pipefail

ACTION="$1"
shift
PIPELINES_PATH="$1"
shift
PIPELINE="$1"
shift

ACTIONS=("run" "schedule")
if [[ ! " ${ACTIONS[@]} " =~ " ${ACTION} " ]] ; then
    echo "Error: Invalid action '${ACTION}', valid values: ${ACTIONS[@]}"
    exit 1
fi

pip install .

export PYTHONPATH=$(cd "${PIPELINES_PATH}/.."; pwd):${PYTHONPATH:-}
deploy ${ACTION} ${PIPELINE}
