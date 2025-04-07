#!/bin/bash

num_nodes=1
num_devices_per_node=1
save_plot=false

while getopts "o:n:d:p" opt; do
    case $opt in
        o) output="$OPTARG" ;;
        n) num_nodes="$OPTARG" ;;
        d) num_devices_per_node="$OPTARG" ;;
        p) save_plot=true ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

if [ "$save_plot" = true ]; then
    opt_args="-p"
else
    opt_args=""
fi

root_dir=$(cd "$(dirname "$0")" && pwd)

docker run --ipc=host --shm-size=200g -it --rm --runtime=nvidia \
    -v ${root_dir}:/workspace/torchcap \
    torchcap-env \
    bash -c "torchrun --nproc_per_node=$num_devices_per_node --nnodes=$num_nodes /workspace/torchcap/torchcap/cluster_env.py -o /workspace/torchcap/configs/$output $opt_args"