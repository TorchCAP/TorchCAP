import os
import math

import torch
from torch.fx import Node
from torch.export.graph_signature import InputKind
from torch.utils._pytree import tree_map_only

from torchcap.fx_utils import size_of


# This value is hard-coded here:
# https://github.com/pytorch/pytorch/blob/5fba5d83f0703ff8077ab65448a998e9ad6598fd/c10/cuda/CUDACachingAllocator.cpp#L117
_PYTORCH_MIN_ALLOCATE = (
    2**9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)


class StorageInfo:
    def __init__(self, size: int, element_size: int, device: torch.device, used_by: set[Node], is_freeable: bool = True):
        self.size = size
        self.element_size = element_size
        self.device = device
        self.used_by = used_by
        self.freeable = is_freeable

    def mem_consumed(self) -> int:
        """
        Calculates the memory consumed by the tensor storage, considering device-specific allocation rules.

        Returns:
            int: The memory consumed in bytes.
        """
        mem = self.size * self.element_size
        if self.device.type == "cuda":
            return math.ceil((mem) / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE
        return mem


def get_storages(node: Node) -> set[torch.UntypedStorage]:
    stors = set()
    tree_map_only(torch.Tensor, lambda x: stors.add(x.untyped_storage()), node.meta["val"])
    return stors


def estimate_memory(mod: torch.nn.Module, gm: torch.fx.GraphModule) -> tuple[dict[str, int], int]:
    nodes = list(gm.graph.nodes)
    node_to_step = {n: i for i, n in enumerate(nodes)}

    memory_breakdown = {"PARAM": 0, "BUFFER": 0, "OPT": 0, "ACT": 0}
    memory_breakdown["PARAM"] = sum([param.numel() * param.element_size() for param in mod.parameters()])
    memory_breakdown["BUFFER"] = sum([buf.numel() * buf.element_size() for buf in mod.buffers()])
    memory_breakdown["ACT"] = sum([size_of(n) for n in nodes if n.op != "placeholder"])

    # extract storage infos from the graph
    stor_infos: dict[torch.UntypedStorage, StorageInfo] = {}
    for node in nodes:
        if "val" not in node.meta:
            continue

        for stor in get_storages(node):
            if stor not in stor_infos:
                # parameters and buffers are not free-able
                is_freeable = node.op != "placeholder"
                stor_infos[stor] = StorageInfo(
                    stor.size(), stor.element_size(), stor.device, 
                    set([node]), is_freeable=is_freeable)
            stor_infos[stor].used_by.add(node)

        for arg in node.args:
            if isinstance(arg, Node):
                for stor in get_storages(arg):
                    assert stor in stor_infos
                    stor_infos[stor].used_by.add(node)

    # analyze the liveness of the storages
    liveness: dict[StorageInfo, tuple[int, int]] = {}
    for info in stor_infos.values():
        start_step = min(node_to_step[n] for n in info.used_by)
        end_step = None
        if info.freeable:
            end_step = max(node_to_step[n] for n in info.used_by)
        liveness[info] = (start_step, end_step)
        # print(f"liveness nodes: {info.used_by}, start: {start_step}, end: {end_step}")

    # update the memory changes at each step
    memory_deltas = [0 for _ in range(len(nodes) + 1)]
    for info, (start, end) in liveness.items():
        memory_deltas[start] += info.mem_consumed()
        if end is not None:
            memory_deltas[end + 1] -= info.mem_consumed()

    # cumulative memory at each step
    memory_used = 0
    peak_memory = 0
    for i, delta in enumerate(memory_deltas):
        # if i < len(nodes):
        #     print(f"Memory at step {i}: {nodes[i].name} delta={delta / 2**30} GiB, memory_used={memory_used / 2**30:.2f} GiB")
        memory_used += delta
        peak_memory = max(peak_memory, memory_used)

    # print(f"[DEBUG] Peak memory: {peak_memory / 2**30} GiB")

    return memory_breakdown, peak_memory
