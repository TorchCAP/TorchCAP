import torch

from torchcap.cluster_env import (
    ClusterEnv,
    profile_device_compute_time,
    profile_device_dram_time_model,
)


def test_cluster_env_profiling():
    import uuid
    cluster_env = ClusterEnv.profile_all(torch.device("cuda"))
    cluster_env.device_compute_time_model[torch.float16].plot(f"float16_compute_time_{uuid.uuid4()}.png")
    cluster_env.device_compute_time_model[torch.float32].plot(f"float32_compute_time_{uuid.uuid4()}.png")
    cluster_env.device_compute_time_model[torch.bfloat16].plot(f"bfloat16_compute_time_{uuid.uuid4()}.png")
    cluster_env.device_dram_time_model.plot(f"dram_time_{uuid.uuid4()}.png")


if __name__ == "__main__":
    test_cluster_env_profiling()
