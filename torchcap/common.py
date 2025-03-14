from torchcap.cluster_env import ClusterEnv
from torch.distributed import DeviceMesh

class torchcapOptions:

    def __init__(self,
        perf_model: str = "roofline",
        cluster_env = ClusterEnv(),
        device_mesh: DeviceMesh = None,
    ):
        self.perf_model = perf_model
        self.cluster_env = cluster_env
        self.device_mesh = device_mesh
