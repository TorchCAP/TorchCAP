# set -ex

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

export PYTHONPATH="$root_dir:$root_dir/third_party/torchtitan"

MODEL_NAME=${1:-"facebook/opt-2.7b"}

mkdir -p ./logs
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
log_file="./logs/test_huggingface_${MODEL_NAME//\//-}_${current_time}.log"
echo "Logging to $log_file"

python $root_dir/examples/huggingface/test_huggingface.py --model ${MODEL_NAME} > $log_file 2>&1
