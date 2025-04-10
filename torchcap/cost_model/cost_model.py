from typing import Any

import operator
import warnings
import torch
from torch import fx
import torch.utils._pytree as pytree
from torch.utils._mode_utils import no_dispatch
from torch._guards import active_fake_mode
from torch.distributed import DeviceMesh

from torchcap.cost_model.comm_model import is_collective
from torchcap.cost_model.flop_counter import FlopCounterMode
from torchcap.cost_model.memory_estimator import MemoryEstimator
from torchcap.fx_utils import materialize_arg, dtypes_of, size_of
from torchcap.common import CAPConfig
from torchcap.cost_model.runtime_estimator import RuntimeEstimator
from torchcap.cost_model import comm_model, memory
from torch._ops import HigherOrderOperator

c10d_ops = torch.ops.c10d
aten = torch.ops.aten


# No fall-back kernel needed/exists for view ops
_VIEW_OPS = {
    aten.lift_fresh,
    aten.t,
    aten.transpose,
    aten.view,
    aten.detach,
    aten._unsafe_view,
    aten.split,
    aten.adjoint,
    aten.as_strided,
    aten.diagonal,
    aten.expand,
    aten.expand_as,
    aten.movedim,
    aten.permute,
    aten.select,
    aten.squeeze,
    aten.mT,
    aten.mH,
    aten.real,
    aten.imag,
    aten.view_as,
    aten.unflatten,
    aten.unfold,
    aten.unbind,
    aten.unsqueeze,
    aten.vsplit,
    aten.hsplit,
    aten.split_with_sizes,
    aten.swapaxes,
    aten.swapdims,
    aten.chunk,
}
# We can ignore benchmarking tensor create ops
_CREATE_OPS = {
    aten.randint,
    aten.randn,
    aten.rand,
    aten.randn_like,
    aten.rand_like,
    aten.randint_like,
    aten.arange,
    aten.ones_like,
    aten.zeros_like,
}

_IGNORE_OPS = _VIEW_OPS | _CREATE_OPS | {
    operator.getitem
}

float_types: set[torch.dtype] = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}

# time_units = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}
time_units = {"ns": 1e-3, "us": 1, "ms": 1e3, "s": 1e6}
memory_units = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30}


def round_time(time: float) -> str:
    for unit, factor in sorted(time_units.items(), key=lambda x: x[1], reverse=True):
        if time >= factor:
            return time / factor, unit
    return time, "ns"


def round_memory(size: int) -> str:
    for unit, factor in sorted(memory_units.items(), key=lambda x: x[1], reverse=True):
        if size >= factor:
            return size / factor, unit
    return size, "B"


class GraphInfo:

    def __init__(
        self,
        all_nodes: list[str],
        all_edges: list[tuple[str, str]],
        all_node_runtimes: dict[str, float],
        all_node_memories: dict[str, float],
        memory_uses: list[float],
        peak_memory: int,
    ):
        self.all_nodes = all_nodes
        self.all_edges = all_edges
        self.all_node_runtimes = all_node_runtimes
        self.all_node_memories = all_node_memories
        self.memory_uses = memory_uses
        self.peak_memory = peak_memory

    def get_total_runtime(self) -> float:
        return float(sum(self.all_node_runtimes.values()))

    def print_tabular(self):
        from tabulate import tabulate
        rows = []
        for node in self.all_nodes:
            rows.append([node, self.all_node_runtimes[node], self.all_node_memories[node]])
        print(tabulate(rows, headers=["Node", "Runtime (ns)", "Memory (B)"]))
        t, unit = round_time(self.get_total_runtime())
        print(f"Total runtime ({unit}): {t:.3f}")
        m, unit = round_memory(self.peak_memory)
        print(f"Peak memory ({unit}): {m:.3f}")
        # p, unit = round_memory(self.memory_uses["PARAM"])
        # print(f"Total parameter size ({unit}): {p:.3f}")
        for mem_type, size in self.memory_uses.items():
            p, unit = round_memory(size)
            print(f"{mem_type} ({unit}): {p:.3f}")


# The runtime estimation models are adapted from https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tools/runtime_estimator.py

# Adapted from: https://github.com/pytorch/pytorch/blob/9b902b3ee3bd608a19543362b66bf06c373dd374/torch/_subclasses/fake_tensor.py#L1969  # noqa: PGH004,B950
# NB: returns fake tensors
def _maybe_run_and_benchmark_fallback_kernel(  # type: ignore[no-untyped-def]
    func,
    args,
    kwargs,
):
    """
    Runs and benchmarks a fallback kernel for a given function.

    Args:
        func (Callable): The function to benchmark.
        args (Tuple): The arguments to pass to the function.
        kwargs (Dict[str, Any]): The keyword arguments to pass to the function.
        orig_not_implemented_exception (Exception): The original exception to raise if the fallback kernel
            is not implemented.

    Returns:
        Tuple[Any, float]: A tuple containing the result of the function and
            the mean operation time in milliseconds.
    """
    # these should all be supported, just to be safe
    # avoid fallback for operators which inplace modify metadata
    # because the input fake tensors would be umodified
    if torch.Tag.inplace_view in func.tags:  # type: ignore[attr-defined]
        raise NotImplementedError

    fake_mode = active_fake_mode()

    inp_impls = {}
    flat_args, args_spec = pytree.tree_flatten((args, kwargs))
    # Don't use in_kernel_invocation_manager(fake_mode) as we want to do
    # REAL compute (not with meta device)
    with no_dispatch():

        def to_real_tensor(e):  # type: ignore[no-untyped-def]
            if fake_mode.is_our_fake(e):
                if e.dtype in float_types:
                    out = torch.rand_like(e, device=e.fake_device)
                else:
                    out = torch.ones_like(e, device=e.fake_device)
                if e.is_sparse:
                    out._coalesced_(e.is_coalesced())
                inp_impls[id(out)] = e
                return out
            return e

        flat_args = [to_real_tensor(a) for a in flat_args]
        args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
        r = func(*args, **kwargs)
        warmup_iters, actual_iters = 2, 3
        for _ in range(warmup_iters):
            func(*args, **kwargs)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(torch.cuda.current_stream())
        for _ in range(actual_iters):
            func(*args, **kwargs)
        end_event.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        cuda_time = start_event.elapsed_time(end_event)
        # mean_op_time = cuda_time / actual_iters * 1e6 # in nanoseconds
        mean_op_time = cuda_time / actual_iters * 1e3 # in microseconds

    storages = set()

    for e in flat_args:
        if isinstance(e, torch.Tensor):
            if not e.is_sparse:
                storages.add(e._typed_storage()._cdata)

    # TODO: also check metadata change on inputs
    # proper aliasing/metadata relationship between outputs and inputs will
    # not be set up, bc of conversion to device, unless we can reuse an
    # input impl

    def map_out(e):  # type: ignore[no-untyped-def]
        if id(e) not in inp_impls and (
            isinstance(e, torch.Tensor)
            and not e.is_sparse
            and e._typed_storage()._cdata in storages
        ):
            raise NotImplementedError

        if isinstance(e, torch.Tensor):
            if id(e) in inp_impls:
                return inp_impls[id(e)]
            else:
                return fake_mode.fake_tensor_converter.from_real_tensor(
                    fake_mode, e
                )
        else:
            return e

    return (pytree.tree_map(map_out, r), mean_op_time)


def estimate_runtime_using_benchmarking(node: fx.Node) -> tuple[Any, float]:  # type: ignore[no-untyped-def]
    """
    Estimates the runtime of a function using benchmarking.

    Args:
        func: The function to estimate.
        args: The arguments to pass to the function.
        kwargs: The keyword arguments to pass to the function.
        res: The result of the function.

    Returns:
        Tuple[Any, float]: A tuple containing the result of the function and
            the mean operation time in milliseconds.
    """
    args, kwargs = pytree.tree_map(materialize_arg, (node.args, node.kwargs))
    func = node.target

    mean_op_time = 0.0
    if func._overloadpacket not in _VIEW_OPS:
        try:
            res, mean_op_time = _maybe_run_and_benchmark_fallback_kernel(
                func,
                args,
                kwargs,
            )
            return mean_op_time
        except NotImplementedError:
            warnings.warn(f"No fallback kernel found for {func._overloadpacket}")
    return mean_op_time


def estimate_runtime_using_roofline(node: fx.Node, options: CAPConfig) -> float:
    # Estimate the FLOPs and compute time
    args, kwargs = pytree.tree_map(materialize_arg, (node.args, node.kwargs))
    with FlopCounterMode(display=False) as mode:
        node.target(*args, **kwargs)
    # We divide by a factor of 2 to get the MACs (multiply and accumulate)
    flop_count = max(mode.get_total_flops() / 2, 1)

    # Estimate the compute time taken to execute the node
    compute_time = 0
    out_dtype = dtypes_of(node).pop()
    if out_dtype in (torch.float16, torch.bfloat16, torch.float32):
        # # This actually gives peta-FLOPs/s hence multiply by 1e15 to get the FLOPs/s
        # peak_gpu_flops = get_device_tflops(out_dtype) * 1e15
        # # We can expect to achieve 75% of theoretical peak flops
        # factor = 0.75
        # peak_empirical_flops = factor * peak_gpu_flops
        # # Multiply by 1e9 to get the time in nanoseconds
        # compute_time = (flop_count / peak_empirical_flops) * 1e9
        compute_time = options.cluster_env.get_device_compute_time(out_dtype, flop_count)

    # Estimate the memory access time taken to execute the node
    # gpu_memory_bandwidth = get_gpu_dram_gbps()
    read_bytes = sum(
        size_of(a) for a in node.args if isinstance(a, fx.Node)
    )
    write_bytes = size_of(node)
    total_bytes = read_bytes + write_bytes
    # The GPU memory bandwidth is in GB/s so the transfer time is in nanoseconds
    transfer_time = options.cluster_env.get_device_dram_time(total_bytes)

    return max(compute_time, transfer_time)


def group_name_to_mesh_dim(group_name: str, device_mesh: DeviceMesh) -> int:
    assert isinstance(group_name, str)
    for dim, info in enumerate(device_mesh._dim_group_infos):
        # dim_group_info = (group_tag, group_ranks, group_name)
        if info[2] == group_name:
            return dim
    raise ValueError(f"Group name {group_name} not found in device mesh")


def estimate_collective_runtime(node: fx.Node, config: CAPConfig) -> float:
    assert is_collective(node)

    if comm_model.is_wait(node):
        return 0.0

    mesh_topo = config.cluster_env.mesh_topo
    op_bytes = comm_model.get_collective_input_size_bytes(node)
    # print(f"[DEBUG] collective: {node.name}, args={node.args}, kwargs={node.kwargs} mesh_dim_names={config.device_mesh._dim_group_infos}")
    mesh_dim = group_name_to_mesh_dim(node.args[-1], config.cluster_env.mesh_topo.get_device_mesh())

    kernel_name = node.target.__name__
    if "all_reduce" in kernel_name:
        t = comm_model.all_reduce_cost(op_bytes, mesh_topo, mesh_dim)
    elif "all_gather" in kernel_name:
        t = comm_model.all_gather_cost(op_bytes, mesh_topo, mesh_dim)
    elif "reduce_scatter" in kernel_name:
        t = comm_model.reduce_scatter_cost(op_bytes, mesh_topo, mesh_dim)
    elif "all_to_all" in kernel_name:
        t = comm_model.all_to_all_cost(op_bytes, mesh_topo, mesh_dim)
    elif "broadcast" in kernel_name:
        t = comm_model.broadcast_cost(op_bytes, mesh_topo, mesh_dim)
    else:
        raise ValueError(f"Unsupported collective kernel: {kernel_name}")
    return t


def is_hops(node: fx.Node) -> bool:
    return isinstance(node.target, HigherOrderOperator)


def estimate_runtime(node: fx.Node, config: CAPConfig) -> float:
    if node.op != "call_function" or node.target in _IGNORE_OPS or is_hops(node):
        return 0

    if is_collective(node):
        try:
            return estimate_collective_runtime(node, config)
        except ValueError as e:
            # We don't know how to estimate runtime for this collective,
            # falling back to 0
            warnings.warn(f"No runtime estimate for collective {node.name}: {e}")
            return 0

    estimator = RuntimeEstimator()
    # Estimate the FLOPs and compute time
    args, kwargs = pytree.tree_map(materialize_arg, (node.args, node.kwargs))
    with estimator(config):
        node.target(*args, **kwargs)
    op_time = estimator.total_runtime

    # measured = estimate_runtime_using_benchmarking(node)
    # def error(a, b):
    #     if b == 0:
    #         return 0
    #     return (a - b) / b * 100
    # print(f"node: {node.name}, target: {node.target}, {op_time=:.2f}, {measured=:.2f}, Error: {op_time - measured:.3f} ({error(op_time, measured):.3f}%)")
    
    hardware_overhead_coefficient = 0.92
    op_time *= hardware_overhead_coefficient


    return op_time


def estimate_memory_using_estimator(program: torch.export.ExportedProgram) -> float:
    with MemoryEstimator() as estimator:
        args, kwargs = program.example_inputs
        estimator.track_externals(
            program.module(), *args, *kwargs.values())
        program.module()(*args, **kwargs)
    return estimator.memory_use, estimator.max_memory


def estimate_graph_cost(mod: torch.nn.Module, gm: torch.fx.GraphModule, config: CAPConfig) -> GraphInfo:
    all_nodes = [node.name for node in gm.graph.nodes]
    all_edges = [
        (node.name, user.name) for node in gm.graph.nodes for user in node.users
    ]
    all_node_runtimes = {node.name: estimate_runtime(node, config) for node in gm.graph.nodes}
    all_node_memories = {node.name: size_of(node) for node in gm.graph.nodes}
    memory_uses, peak_memory = memory.estimate_memory(mod, gm)



    return GraphInfo(
        all_nodes=all_nodes,
        all_edges=all_edges,
        all_node_runtimes=all_node_runtimes,
        all_node_memories=all_node_memories,
        memory_uses=memory_uses,
        peak_memory=peak_memory,
    )
