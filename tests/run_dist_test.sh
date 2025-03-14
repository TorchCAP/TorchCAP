#!/bin/bash

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Add project root to PYTHONPATH
export PYTHONPATH="$root_dir:$root_dir/third_party/torchtitan"

# Run pytest with specified test file
torchrun --nproc_per_node=2 pytest -v -s "$@"
