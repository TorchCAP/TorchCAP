set -ex

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

export PYTHONPATH="$root_dir:$root_dir/third_party/torchtitan"

MODEL_TYPE=${1:-"debug_model"}

CONFIG_FILE="$root_dir/third_party/torchtitan/train_configs/${MODEL_TYPE}.toml"

python $root_dir/examples/torchtitan/train.py --job.config_file ${CONFIG_FILE}
