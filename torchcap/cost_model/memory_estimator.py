import weakref
import functools
from enum import Enum, auto
import os
import math

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only, tree_flatten


# This value is hard-coded here:
# https://github.com/pytorch/pytorch/blob/5fba5d83f0703ff8077ab65448a998e9ad6598fd/c10/cuda/CUDACachingAllocator.cpp#L117
_PYTORCH_MIN_ALLOCATE = (
    2**9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)


class MemType(Enum):
    PARAM = auto()
    BUFFER = auto()
    ACT = auto()
    GRAD = auto()
    OPT = auto()


class MemoryEstimator(TorchDispatchMode):
    def __init__(self):
        # counter of storage ids to live references
        self.live_storages = weakref.WeakKeyDictionary()
        self.curr_memory = 0
        self.max_memory = 0
        self.memory_use = {
            MemType.PARAM: 0,
            MemType.BUFFER: 0,
            MemType.ACT: 0,
            MemType.GRAD: 0,
            MemType.OPT: 0,
        }

    @property
    def is_bw(self):
        """
        A boolean marking if this is currently running during the backward pass or not
        """
        return torch._C._current_graph_task_id() != -1

    def _track_model(self, model: torch.nn.Module):
        for param in model.parameters():
            self.track_tensor_memory_use(param, MemType.PARAM)

    def _track_optimizer_states(self, optimizer: torch.optim.Optimizer):
        for states in optimizer.state.values():
            for val in states.values():
                if isinstance(val, torch.Tensor):
                    self.track_tensor_memory_use(val, MemType.OPT)

    def track_externals(self, *externals):
        flat_ext, _ = tree_flatten(externals)
        for obj in flat_ext:
            if isinstance(obj, torch.nn.Module):
                self._track_model(obj)
            elif isinstance(obj, torch.Tensor):
                self.track_tensor_memory_use(obj, MemType.ACT)
            elif isinstance(obj, torch.optim.Optimizer):
                self._track_optimizer_states(obj)
            else:
                raise ValueError(f"Unrecognized object: {type(obj)}")

    def track_tensor_memory_use(self, tensor: torch.Tensor, mem_type: MemType):
        # already accounted for
        stor = tensor.untyped_storage()
        if stor in self.live_storages:
            return

        self.live_storages[stor] = True
        nbytes = tensor.untyped_storage().nbytes()
        mem_consumed = 0
        if tensor.device.type == "cuda":
            mem_consumed = math.ceil((nbytes) / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE

        self.memory_use[mem_type] += mem_consumed

        # new storage, add to memory
        self.change_memory(mem_consumed)

        # when this storage dies, we need to adjust memory
        weakref.finalize(stor, functools.partial(self.change_memory, -mem_consumed))

    def change_memory(self, delta):
        self.curr_memory += delta
        self.max_memory = max(self.curr_memory, self.max_memory)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        kwargs = kwargs if kwargs is not None else {}
        out = func(*args, **kwargs)
        if self.is_bw:
            tree_map_only(torch.Tensor,
                functools.partial(self.track_tensor_memory_use, mem_type=MemType.GRAD), out)
        else:
            tree_map_only(torch.Tensor,
                functools.partial(self.track_tensor_memory_use, mem_type=MemType.ACT), out)
        return out