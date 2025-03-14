root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

CUDA_DEVICES="all"

export PYTHONPATH="$root_dir:$root_dir/third_party/torchtitan"

TEST_TYPE="test_single_gpu"
MODEL_NAME="facebook/opt-2.7b"
CLUSTER_ENV="a5000_24g_gala1.json"

WORLD_SIZE=1

while getopts ":t:m:e:" opt; do
    case ${opt} in
        t ) TEST_TYPE=${OPTARG} ;;
        m ) MODEL_NAME=${OPTARG} ;;
        e ) CLUSTER_ENV=${OPTARG} ;;
    esac
done

mkdir -p ./logs
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
log_file="./logs/test_huggingface_${MODEL_NAME//\//-}_${current_time}.log"
echo "Logging to $log_file"

# python $root_dir/examples/huggingface/test_huggingface.py --model ${MODEL_NAME} > $log_file 2>&1

if [ "$TEST_TYPE" == "test_single_gpu" ]; then
cmd="python /workspace/torchcap/examples/huggingface/test_single_gpu.py \
 --model ${MODEL_NAME} \
 --cluster-env /workspace/torchcap/configs/${CLUSTER_ENV}"
elif [ "$TEST_TYPE" == "test_multi_gpu" ]; then
cmd="torchrun --nproc_per_node=$WORLD_SIZE \
 examples/huggingface/test_multi_gpu.py \
 --model ${MODEL_NAME} \
 --cluster-env configs/${CLUSTER_ENV}"
fi

echo "Running $cmd"

sudo docker run \
    --ipc=host --shm-size=200g -it --rm --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_DEVICES \
    -v ${root_dir}:/workspace/torchcap \
    -v $HF_HOME:/workspace/cache \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e HF_HOME=/workspace/cache \
    -e PYTHONPATH=$PYTHON_PATH:/workspace/torchcap \
    torchcap-env \
    bash -c "nvidia-smi; $cmd" > $log_file 2>&1
