# TorchCAP

## Setup

```bash
docker build -t torchcap-env .
```

## Running Huggingface models

```bash
bash examples/huggingface/run_docker.sh -m facebook/opt-2.7b -e a5000_24g_gala1.json
```

## Profiling hardware

```bash
bash profile_cluster.sh -o profile.json
```