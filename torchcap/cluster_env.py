from typing import Optional, Any
from functools import cmp_to_key
import gc
import warnings
import numpy as np
from numpy.typing import NDArray, ArrayLike
import matplotlib.pyplot as plt
import torch.cuda.profiler as profiler
import math
import time
from collections import defaultdict
from functools import partial
import argparse
import os
import uuid

import torch
from torch._inductor.utils import (
    get_gpu_dram_gbps as get_peak_gpu_dram_gbps,
)
from torch.distributed import DeviceMesh
from torch.distributed.device_mesh import _get_device_handle
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

float_types: set[torch.dtype] = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]


str_to_dtype = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
}


def print_rank(msg: str):
    if dist.is_initialized():
        print(f"[Rank {dist.get_rank()}] {msg}")
    else:
        print(f"[Rank 0] {msg}")


def max_clock_rate_khz():
    from triton.testing import nvsmi
    return nvsmi(["clocks.max.sm"])[0] * 1e3 # ghz -> khz


def get_max_tensorcore_tflops(dtype, clock_rate, device=None):
    import torch

    if not device:
        device = torch.cuda.current_device()

    num_tensor_cores = torch.cuda.get_device_properties(device).multi_processor_count * 4
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8:
        assert dtype == torch.float16
        ops_per_sub_core = 256  # 2 4x4x4 Tensor Cores
    else:
        if dtype in [torch.float32, torch.int32]:
            ops_per_sub_core = 256
        elif dtype in [torch.float16, torch.bfloat16, torch.int16]:
            ops_per_sub_core = 512
        elif dtype in [torch.int8]:
            ops_per_sub_core = 1024
        else:
            raise RuntimeError("dtype not supported")
    tflops = num_tensor_cores * clock_rate * ops_per_sub_core * 1e-9
    return tflops


def get_max_simd_tflops(dtype, clock_rate, device=None):
    import torch

    if not device:
        device = torch.cuda.current_device()

    num_cuda_cores = torch.cuda.get_device_properties(device).multi_processor_count * 128 # ampere
    if dtype in [torch.float32, torch.float16, torch.bfloat16]:
        ops_per_core = 2
    elif dtype in [torch.int32]:
        ops_per_core = 1
    tflops = num_cuda_cores * clock_rate * ops_per_core * 1e-9
    return tflops


def get_peak_gpu_tflops(dtype):
    assert dtype in (torch.float16, torch.bfloat16, torch.float32)

    sm_clock = max_clock_rate_khz()
    if dtype in (torch.float16, torch.bfloat16):
        return get_max_tensorcore_tflops(dtype, sm_clock)

    if torch.backends.cuda.matmul.allow_tf32:
        return get_max_tensorcore_tflops(torch.float32, sm_clock)
    else:
        return get_max_simd_tflops(torch.float32, sm_clock)


class AlphaBetaModel:

    def __init__(self, alpha: NDArray, beta: NDArray, breakpoints: NDArray, data: tuple[NDArray, NDArray]):
        self.alpha = alpha
        self.beta = beta
        self.breakpoints = breakpoints
        self.data = data

    def __str__(self):
        return f"AlphaBetaModel(alpha={self.alpha}, beta={self.beta}, breakpoints={self.breakpoints})"

    def __repr__(self):
        return self.__str__()

    def __call__(self, x: float) -> float:
        if x == 0:
            return 0.0
        elif x < 0:
            raise ValueError("x must be non-negative")
        
        breakpoints = self.breakpoints
        breakpoints = breakpoints[:-1]  # Exclude the last breakpoint
        breakpoints[0] = -np.inf        # Set the first point to be 0

        bkID = 0
        for i, e in enumerate(breakpoints):
            if x > e:
                bkID = i
            else:
                break
        alpha = self.alpha[bkID]
        beta = self.beta[bkID]

        return max(0, alpha + beta * x)

    @staticmethod
    def from_data(x: NDArray, y: NDArray, **kwargs):
        import pwlf
        model = pwlf.PiecewiseLinFit(x, y)

        num_seg = 6
        id = np.linspace(0, len(x) - 1, num_seg + 1).astype(int)
        breakpoints = np.array(x[id], dtype=np.float32)
        model.fit_with_breaks(breakpoints)

        intercepts = model.intercepts
        slopes = model.slopes
        breakpoints = model.fit_breaks
        print(f"Created alpha-beta model")
        print(f"  - intercept: {intercepts}")
        print(f"  - slope: {slopes}")
        print(f"  - breakpoints: {breakpoints}")

        return AlphaBetaModel(intercepts, slopes, breakpoints, (x, y))

        # from sklearn.linear_model import LinearRegression

        # model = LinearRegression(**kwargs)
        # model.fit(x.reshape(-1, 1), y.reshape(-1, 1))

        # # Obtain the slopes, intercepts and breakpoints of the fitted piecewise linear functions
        # intercept = model.intercept_[0]
        # slope = model.coef_[0][0]

        # print(f"Created alpha-beta model")
        # print(f"  - intercept: {intercept}")
        # print(f"  - slope: {slope}")

        # return AlphaBetaModel(intercept, slope, (x, y))

    def to_dict(self):
        return {
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
            "breakpoints": self.breakpoints.tolist(),
            "data": (self.data[0].tolist(), self.data[1].tolist()),
        }

    @staticmethod
    def from_dict(data: dict):
        x = np.array(data["data"][0])
        y = np.array(data["data"][1])

        # Create a piecewise linear fit model
        import pwlf
        model = pwlf.PiecewiseLinFit(x, y)
        
        # evenly pick points of x as breakpoints
        num_seg = 6
        id = np.linspace(0, len(x) - 1, num_seg + 1).astype(int)
        breakpoints = np.array(x[id], dtype=np.float32)
        model.fit_with_breaks(breakpoints)

        return AlphaBetaModel(
            model.intercepts,
            model.slopes,
            model.fit_breaks,
            (np.array(data["data"][0]), np.array(data["data"][1]))
        )

    def plot(self, file_name: str):
        import matplotlib.pyplot as plt
        import numpy as np

        # Generate points for plotting the piecewise linear fit
        x_hat = np.linspace(np.floor(self.data[0].min()), np.ceil(self.data[0].max()*1.2), 10000)
        y_hat = np.array([self(x_hat[i]) for i in range(len(x_hat))])

        plt.clf()  # Clear any existing plots

        # Plot original data points
        plt.plot(self.data[0], self.data[1], 'o', markersize=8, markerfacecolor='red', label='Data points')
        # Plot piecewise linear fit
        plt.plot(x_hat, y_hat, 'b-', linewidth=2, label='Model')

        plt.legend()
        # plt.xscale('log')  # Using log scale for better visualization   
        # plt.yscale('log')
        plt.grid(True)

        if file_name:
            # Save the plot
            plt.savefig(file_name)
            plt.close()  # Close the plot to free memory
            print(f"Saved plot to {file_name}")


def get_shape(x: Any) -> tuple[int, ...]:
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    elif isinstance(x, np.ndarray):
        return x.shape
    elif isinstance(x, (list, tuple)):  # Assuming it's a nested list or tuple
        try:
            return np.shape(x)
        except:
            return (len(x),)
    else:
        raise TypeError("Unsupported type. Must be torch.Tensor, np.ndarray, or a nested list/tuple.")


class MeshTopology:
    def __init__(
        self,
        device_type: str,
        mesh_shape: tuple[int, ...],
        mesh_dim_names: tuple[str, ...],
        comm_model: dict[int, AlphaBetaModel],
    ):
        self.device_type = device_type
        self.logical_mesh_shape = mesh_shape
        self.physical_mesh_shape: Optional[tuple[int, ...]] = None
        self.mesh_dim_names = mesh_dim_names
        self.comm_model: dict[int, AlphaBetaModel] = comm_model
        self.device_mesh: Optional[DeviceMesh] = None

    def __str__(self):
        return (
            "MeshTopology(\n"
            f"    device_type={self.device_type},\n"
            f"    logical_mesh_shape={self.logical_mesh_shape},\n"
            f"    physical_mesh_shape={self.physical_mesh_shape},\n"
            f"    mesh_dim_names={self.mesh_dim_names},\n"
            f"    comm_model={self.comm_model}\n"
            ")"
        )

    def __repr__(self):
        return self.__str__()

    def materialize(self, physical_mesh: ArrayLike | torch.Tensor):
        self.physical_mesh_shape = get_shape(physical_mesh)
        self.device_mesh = DeviceMesh(self.device_type, physical_mesh, mesh_dim_names=self.mesh_dim_names)

    @property
    def mesh_shape(self) -> tuple[int, ...]:
        return self.physical_mesh_shape if self.physical_mesh_shape is not None else self.logical_mesh_shape

    @property
    def world_size(self) -> int:
        return np.prod(self.physical_mesh_shape) if self.physical_mesh_shape is not None else np.prod(self.logical_mesh_shape)

    def get_device_mesh(self) -> DeviceMesh:
        assert self.device_mesh is not None
        return self.device_mesh


def profile_device_compute_time():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = np.round(np.linspace(2**7, 2**15, num=30)).astype(int)
    # Ensure all are even. Odd number sizes are rarely used in practice and will lead to poor throughput.
    inputs[inputs % 2 != 0] += 1

    device_compute_time_model = {}

    for dtype in float_types:
        op_flops = []
        elapsed_times = []
        achieved_tflops = [] # for debug

        for N in inputs:
            A = torch.randn((N, N), device=device, dtype=dtype)
            B = torch.randn((N, N), device=device, dtype=dtype)

            warmup = 3
            repeat = 10

            # Warmup CUDA to avoid initial overhead
            for _ in range(warmup):
                torch.mm(A, B)
            torch.cuda.synchronize()

            # Measure execution time
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(repeat):
                torch.mm(A, B)
            end.record()

            # Synchronize to get accurate timing
            torch.cuda.synchronize()
            elapsed_time_ms = start.elapsed_time(end) / repeat # in milliseconds

            # Compute FLOPs for matrix multiplication: 2*N^3
            flops = 2 * N**3

            # Store results
            op_flops.append(flops)
            elapsed_times.append(elapsed_time_ms * 1e3) # ms -> us
            achieved_tflops.append(float(flops) / elapsed_time_ms * 1e-9) # kflop/s -> tflop/s

        # Convert to numpy arrays
        op_flops = np.array(op_flops)
        elapsed_times = np.array(elapsed_times)

        print(f"dtype: {dtype}")
        print(f"results: {list(zip(op_flops.tolist(), elapsed_times.tolist(), achieved_tflops))}")

        device_compute_time_model[dtype] = AlphaBetaModel.from_data(op_flops, elapsed_times)

    return device_compute_time_model


def profile_device_memory_time():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = np.round(np.linspace(2**7, 2**15, num=30)).astype(int)

    device_memory_time_model = {}

    op_bytes = []
    elapsed_times = []
    achieved_gbps = [] # for debug

    for N in inputs:
        A = torch.randn((N, N), device=device, dtype=torch.float32)
        B = torch.randn((N, N), device=device, dtype=torch.float32)

        warmup = 3
        repeat = 20

        # Warmup CUDA to avoid initial overhead
        for _ in range(warmup):
            A.copy_(B, non_blocking=True)
            B.copy_(A, non_blocking=True)
        torch.cuda.synchronize()

        # Measure execution time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(repeat):
            A.copy_(B, non_blocking=True)
            B.copy_(A, non_blocking=True)
        end.record()

        # Synchronize to get accurate timing
        torch.cuda.synchronize()
        elapsed_time_ms = start.elapsed_time(end) / repeat # in milliseconds

        # 2*(A+B) * 4 bytes (fp32)
        nbytes = 8 * N**2 * 4

        # Store results
        op_bytes.append(nbytes)
        elapsed_times.append(elapsed_time_ms * 1e3) # ms -> us
        achieved_gbps.append(float(nbytes) / elapsed_time_ms * 1e-6) # B/ms -> GB/s

    # Convert to numpy arrays
    op_bytes = np.array(op_bytes)
    elapsed_times = np.array(elapsed_times)

    print(f"results: {list(zip(op_bytes.tolist(), elapsed_times.tolist(), achieved_gbps))}")

    device_memory_time_model = AlphaBetaModel.from_data(op_bytes, elapsed_times)

    return device_memory_time_model


def _get_device_from_mesh(mesh: DeviceMesh) -> torch.device:
    if mesh.device_type == "cpu":
        return torch.device("cpu")
    device_handle = _get_device_handle(mesh.device_type)
    return torch.device(mesh.device_type, device_handle.current_device())


def profile_communication_time(device_mesh: DeviceMesh) -> MeshTopology:
    input_bytes = [2**n for n in range(8, 31)]

    warmup = 5
    repeat = 20

    def get_tensor(nbytes: int):
        return torch.ones((nbytes // 4,), device=device, dtype=torch.float32)

    def get_comm_func(op: str, nbytes: int, group: dist.ProcessGroup):
        p = dist.get_world_size(group)
        s = nbytes / 2**30 # size in GB
        if op == "all_reduce":
            input_tensor = get_tensor(nbytes)
            comm_size_gb = 2*(p - 1) / p * s
            func = partial(dist.all_reduce, input_tensor, group=group)
        elif op == "all_gather":
            input_tensor = get_tensor(nbytes // p)
            output_tensor = get_tensor(nbytes)
            comm_size_gb = (p - 1) / p * s
            func = partial(dist.all_gather_into_tensor, output_tensor, input_tensor, group=group)
        elif op == "reduce_scatter":
            input_tensor = get_tensor(nbytes)
            output_tensor = get_tensor(nbytes // p)
            comm_size_gb = (p - 1) / p * s
            func = partial(dist.reduce_scatter_tensor, output_tensor, input_tensor, group=group)
        elif op == "all_to_all":
            input_tensor = get_tensor(nbytes)
            output_tensor = get_tensor(nbytes)
            comm_size_gb = (p - 1) / p * s
            return partial(dist.all_to_all_single, output_tensor, input_tensor, group=group)
        else:
            raise ValueError(f"Invalid communication operation: {op}")
        return func, comm_size_gb

    def profile(group: dist.ProcessGroup, comm_type: str):
        torch.cuda.empty_cache()
        gc.collect()

        torch.cuda.synchronize()
        p = dist.get_world_size(group)

        results = []

        for N in input_bytes:
            comm_func, comm_size_gb = get_comm_func(comm_type, N, group)
            torch.cuda.synchronize()

            # warmup
            for _ in range(warmup):
                comm_func()
            torch.cuda.synchronize()
            dist.barrier(group=group)

            # repeat
            start = time.perf_counter()
            for _ in range(repeat):
                comm_func()
            torch.cuda.synchronize()
            dist.barrier(group=group)
            end = time.perf_counter()

            if device_mesh.get_rank() == 0:
                elasped_us = (end - start) / repeat * 1e6
                results.append((comm_size_gb, elasped_us))

        return results

    comm_model = {}

    # Profile along each mesh dimension
    for mesh_dim in range(device_mesh.ndim):
        group = device_mesh.get_group(mesh_dim=mesh_dim)
        device = _get_device_from_mesh(device_mesh)

        results = []
        for comm_type in ["all_reduce", "all_gather", "reduce_scatter"]:
            res = profile(group, comm_type)
            results.extend(res)

        if device_mesh.get_rank() == 0:
            x, t = zip(*results)
            x = np.array(x)
            t = np.array(t)
            model = AlphaBetaModel.from_data(x, t)
            comm_model[mesh_dim] = model

    mesh_topo = MeshTopology(
        device_type=device_mesh.device_type,
        mesh_shape=device_mesh.shape,
        mesh_dim_names=device_mesh.mesh_dim_names,
        comm_model=comm_model,
    )

    return mesh_topo


class ClusterEnv:
    def __init__(self):
        self.device_name = ""
        self.device_compute_time_model: dict[torch.dtype, Optional[AlphaBetaModel]] = {
            torch.float16: None,
            torch.bfloat16: None,
            torch.float32: None,
            torch.float64: None,
        }
        self.device_memory_time_model: Optional[AlphaBetaModel] = None
        self.mesh_topo: Optional[MeshTopology] = MeshTopology(
            device_type="", mesh_shape=(), mesh_dim_names=(), comm_model={})

    def __str__(self):
        return f"ClusterEnv(\n" \
            f"  device_name={self.device_name},\n" \
            f"  device_compute_time_model={self.device_compute_time_model},\n" \
            f"  device_memory_time_model={self.device_memory_time_model},\n" \
            f"  mesh_topo={self.mesh_topo}\n" \
            f")"

    def __repr__(self):
        return self.__str__()

    def save(self, file_path: str):
        import json
        print(f"Saving cluster environment to {file_path}")
        with open(file_path, "w+") as f:
            json.dump({
                "device_name": self.device_name,
                "device_compute_time_model": {
                    str(dtype): model.to_dict() if model is not None else None
                        for dtype, model in self.device_compute_time_model.items()
                },
                "device_memory_time_model": self.device_memory_time_model.to_dict() 
                    if self.device_memory_time_model is not None else None,
                "mesh_topo": {
                    "device_type": self.mesh_topo.device_type,
                    "mesh_shape": self.mesh_topo.mesh_shape,
                    "mesh_dim_names": self.mesh_topo.mesh_dim_names,
                    "comm_model": {
                        str(dim): model.to_dict() for dim, model in self.mesh_topo.comm_model.items() if model is not None
                    },
                },
            }, f, indent=4)

    @staticmethod
    def from_json(file_path: str):
        import json
        print(f"Loading cluster environment from {file_path} ...")
        cluster_env = ClusterEnv()
        with open(file_path, "r") as f:
            data = json.load(f)
            cluster_env.device_name = data["device_name"]
            cluster_env.device_compute_time_model = {
                str_to_dtype[dtype]: AlphaBetaModel.from_dict(model) if model is not None else None
                    for dtype, model in data["device_compute_time_model"].items()
            }
            cluster_env.device_memory_time_model = AlphaBetaModel.from_dict(data["device_memory_time_model"])
            if "mesh_topo" in data:
                cluster_env.mesh_topo = MeshTopology(
                    device_type=data["mesh_topo"]["device_type"],
                    mesh_shape=data["mesh_topo"]["mesh_shape"],
                    mesh_dim_names=data["mesh_topo"]["mesh_dim_names"],
                    comm_model={int(dim): 
                        AlphaBetaModel.from_dict(model_data) 
                        for dim, model_data in data["mesh_topo"]["comm_model"].items()},
                )
        return cluster_env

    @staticmethod
    def get_device_memory_capacity_bytes() -> int:
        return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory

    def get_device_compute_time(self, dtype: torch.dtype, op_flops: int) -> float:
        if self.device_compute_time_model[dtype] is not None:
            return self.device_compute_time_model[dtype](op_flops)
        warnings.warn(f"No compute time model found for {dtype}. Using peak GPU TFLOPs for estimation")
        return op_flops / (get_peak_gpu_tflops(dtype) * 1000) * 0.7

    def get_device_memory_time(self, op_bytes: int) -> float:
        if self.device_memory_time_model is not None:
            return self.device_memory_time_model(op_bytes)
        warnings.warn(f"No DRAM time model found. Using peak GPU DRAM bandwidth for estimation")
        return op_bytes / get_peak_gpu_dram_gbps() * 0.7

    @staticmethod
    def profile_communication(device_mesh: DeviceMesh):
        cluster_env = ClusterEnv()
        cluster_env.mesh_topo = profile_communication_time(device_mesh)
        return cluster_env

    @staticmethod
    def profile_compute():
        cluster_env = ClusterEnv()
        cluster_env.device_compute_time_model = profile_device_compute_time()
        return cluster_env

    @staticmethod
    def profile_memory():
        cluster_env = ClusterEnv()
        cluster_env.device_memory_time_model = profile_device_memory_time()
        return cluster_env

    @staticmethod
    def profile_all(device_mesh: DeviceMesh):
        cluster_env = ClusterEnv()

        # Only rank 0 should profile the cost of a single device
        if device_mesh.get_rank() == 0:
            print_rank(f"Profiling device {torch.cuda.get_device_name()}")
            cluster_env.device_name = torch.cuda.get_device_name()
            cluster_env.device_compute_time_model = profile_device_compute_time()
            cluster_env.device_memory_time_model = profile_device_memory_time()

        cluster_env.mesh_topo = profile_communication_time(device_mesh)

        return cluster_env



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str, default="all", choices=["all", "compute", "memory", "comm"])
    parser.add_argument("-o", "--output", type=str, default=f"cluster_{str(uuid.uuid4())}.json")
    parser.add_argument("-p", "--save-plot", action="store_true")
    args = parser.parse_args()


    def sort_device(x, y):
        x_free, x_total = torch.cuda.mem_get_info(x)
        y_free, y_total = torch.cuda.mem_get_info(y)
        x_util = (x_total - x_free) / x_total
        y_util = (y_total - y_free) / y_total
        if x_util == y_util:
            return 0
        elif x_util < y_util:
            return -1
        else:
            return 1
    deviceIDs = np.arange(torch.cuda.device_count())
    deviceIDs = sorted(deviceIDs, key=cmp_to_key(sort_device))
    torch.cuda.set_device(int(deviceIDs[0]))

    # torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    num_devices_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
    num_nodes = int(os.environ["WORLD_SIZE"]) // num_devices_per_node
    mesh_shape = (num_nodes, num_devices_per_node) if num_nodes > 1 else (num_devices_per_node,)
    mesh = init_device_mesh("cuda", mesh_shape)
    print(f"Device Mesh: {mesh}")

    if args.type == "all":
        cluster = ClusterEnv.profile_all(device_mesh=mesh)
    elif args.type == "compute":
        cluster = ClusterEnv.profile_compute()
    elif args.type == "memory":
        cluster = ClusterEnv.profile_memory()
    elif args.type == "comm":
        cluster = ClusterEnv.profile_communication(device_mesh=mesh)
    cluster.save(args.output)

    print(f"Done profiling cluster environment ({args.output})")

    if args.save_plot and mesh.get_rank() == 0:
        base_name = os.path.splitext(args.output)[0]
        cluster.device_compute_time_model[torch.float16].plot(base_name + "_compute_time_fp16.png")
        cluster.device_compute_time_model[torch.bfloat16].plot(base_name + "_compute_time_bf16.png")
        cluster.device_compute_time_model[torch.float32].plot(base_name + "_compute_time_fp32.png")
        cluster.device_memory_time_model.plot(base_name + "_memory_time.png")
        for mesh_dim, comm_model in cluster.mesh_topo.comm_model.items():   
            comm_model.plot(base_name + f"_comm_time_dim{mesh_dim}.png")
