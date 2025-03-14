#!/bin/bash

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Add project root to PYTHONPATH
export PYTHONPATH="$root_dir:$root_dir/third_party/torchtitan"

# Run pytest with specified test file
pytest -v -s "$@"
