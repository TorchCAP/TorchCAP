root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

export PYTHONPATH="$root_dir:$root_dir/third_party/torchtitan"

MODEL_NAME="facebook/opt-2.7b"
CLUSTER_ENV="configs/a5000_24g_gala1.json"

while getopts ":n:m:e:" opt; do
    case ${opt} in
        n ) WORLD_SIZE=${OPTARG} ;;
        m ) MODEL_NAME=${OPTARG} ;;
        e ) CLUSTER_ENV=${OPTARG} ;;
    esac
done

mkdir -p ./logs
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
log_file="./logs/test_huggingface_${MODEL_NAME//\//-}_${current_time}.log"
echo "Logging to $log_file"

# python $root_dir/examples/huggingface/test_huggingface.py --model ${MODEL_NAME} > $log_file 2>&1

if [ "$WORLD_SIZE" == "1" ]; then
cmd="python /workspace/torchcap/examples/huggingface/test_single_gpu.py \
 --model ${MODEL_NAME} \
 --cluster-env /workspace/torchcap/${CLUSTER_ENV}"
elif [ "$WORLD_SIZE" -gt "1" ]; then
cmd="torchrun --nproc_per_node=$WORLD_SIZE \
 /workspace/torchcap/examples/huggingface/test_multi_gpu.py \
 --model ${MODEL_NAME} \
 --cluster-env /workspace/torchcap/${CLUSTER_ENV}"
fi

# Check if HF_HOME is set and the directory exists
echo "HF_HOME: $HF_HOME"
if [[ -n "$HF_HOME" && -d "$HF_HOME" ]]; then
  HF_MOUNT="-v $HF_HOME:$HF_HOME"
else
  HF_MOUNT=""
fi

echo "Running $cmd"

docker run \
    --ipc=host --shm-size=200g -it --rm --runtime=nvidia --gpus '"device=1,2"' \
    -v ${root_dir}:/workspace/torchcap \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e PYTHONPATH=$PYTHON_PATH:/workspace/torchcap \
    $HF_MOUNT \
    -e HF_HOME="$HF_HOME" \
    torchcap-env \
    bash -c "$cmd" > $log_file 2>&1
