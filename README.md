# TorchCAP

## Setup

Build the Docker environment required to run TorchCAP:

```bash
docker build -t torchcap-env .
```

## Profiling hardware

TorchCAP can optionally profile your hardware to collect detailed performance metrics. This enables more accurate cost estimation.

Note: If this step is skipped, TorchCAP will fall back on querying system APIs, which may be less precise.

To profile your cluster:

```bash
bash profile_cluster.sh -o <output_file>.json -n <num_nodes> -d <num_devices_per_node> [-p]
```

This command generates a JSON file containing a cost model specific to your hardware. `-p` is an optional option that generates the plots of the cost models. 

## Running TorchCAP with Huggingface models

To run TorchCAP with a specified hardware profile and Huggingface models:

### Single GPU

```
bash examples/huggingface/run_docker.sh -m facebook/opt-6.7b -e profile.json -n 1
```

### Multiple GPUs

```bash
bash examples/huggingface/run_docker.sh -m facebook/opt-6.7b -e profile.json -n 2
```

### Arguments

- ``-n``: Number of GPUs
- ``-m``: Hugging Face model path (e.g., ``facebook/opt-6.7b``)
- ``-e``: Path to the hardware profile JSON file generated during profiling

