#!/bin/bash

set -euo pipefail

pip install -r requirements.txt
pip install -r dev-requirements.txt
pip install ../model_package

cd ..
python -m pytest "$@"
