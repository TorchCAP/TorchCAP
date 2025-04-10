from torchcap.cluster_env import ClusterEnv

class CAPConfig:

    def __init__(self,
        perf_model: str = "roofline",
        cluster_env = ClusterEnv(),
    ):
        self.perf_model = perf_model
        self.cluster_env = cluster_env
