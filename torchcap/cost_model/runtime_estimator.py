# Adapted from https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tools/runtime_estimator.py

# Owner(s): ["module: unknown"]
import math
import os
from collections import defaultdict
from typing import Any, Callable
from typing_extensions import Self

import torch
import torch.utils._pytree as pytree
from torch._guards import active_fake_mode
from torch._inductor.utils import get_device_tflops, get_gpu_dram_gbps
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.mod_tracker import ModTracker
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
# from torch.utils.flop_counter import flop_registry
from torchcap.cost_model.flop_counter import flop_registry
from torchcap.common import torchcapOptions

aten = torch.ops.aten
prim = torch.ops.prim

# This value is hard-coded here:
# https://github.com/pytorch/pytorch/blob/5fba5d83f0703ff8077ab65448a998e9ad6598fd/c10/cuda/CUDACachingAllocator.cpp#L117
_PYTORCH_MIN_ALLOCATE = (
    2**9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)

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
    prim.device,
    aten.slice,
}

__all__ = ["RuntimeEstimator"]


class RuntimeEstimator(TorchDispatchMode):
    """
    Estimates the GPU runtime in milliseconds using various estimation methods under the ``FakeTensorMode``.

    This class provides a ``TorchDispatchMode`` based context manager that can be used to estimate the eager
    runtime of PyTorch functions. It supports two estimation modes, benchmarking (`operator-level-benchmark`) and
    roofline cost modeling (`operator-level-cost-model`).
    For modules executed under this context manager, it agggregates the forward and backward operation runtimes
    and also records their execution orders.

    Attributes:
        mod_runtimes (Dict[str, Dict[str, float]]): A dictionary of module runtimes. The key to the outer dictionary
            is the fully qualified name (FQN) of the module. For each module the forward and backward runtimes of the
            operations are aggregated in the inner dictionary keyed by 'fw' and 'bw'.
        mod_fw_pre_order (List[str]): List of module FQNs in pre-forward execution order.
        mod_bw_pre_order (List[str]): List of module FQNs in pre-backward execution order.
        mod_fw_post_order (List[str]): List of module FQNs in post-forward execution order.
        mod_bw_post_order (List[str]): List of module FQNs in post-backward execution order.
        total_runtime (float): The total estimated runtime in milliseconds.

    Note:
        1) The benchmarking estimate mode will execute kernels on GPU and assumes that every operation can run in
            isolation without causing an OOM error. It is also designed to be used only under ``FakeTensorMode``.
        2) Currently wrapper tensor sub-classes such as ``DTensor`` won't produce correct estimates. We plan to support
            them in future PRs.
        3) We only estimate the compute time, if your code has communication, it will not be considered. Again, we will
            support this in future PRs.

    Example usage:

        .. code-block:: python

            runtime_estimator = RuntimeEstimator()
            with FakeTensorMode():
                module = ...
                optimizer = ...
                inp = ...
                with runtime_estimator(estimate_mode_type="operator-level-cost-model"):
                    loss = module(inp)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                runtime_estimator.display_modulewise_stats()
    """

    _float_types: set[torch.dtype] = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    _no_fallback_kernel: set[torch._ops._OpNamespace] = set()
    fake_mode: FakeTensorMode

    def __init__(self) -> None:
        super().__init__()
        self._estimate: Callable
        self._estimate_mode_type: str
        # self._mod_tracker = ModTracker()
        # self.mod_runtimes: dict[str, dict[str, float]] = defaultdict(
        #     lambda: defaultdict(lambda: 0.0)
        # )
        # self.mod_fw_pre_order: list[str] = []
        # self.mod_bw_pre_order: list[str] = []
        # self.mod_fw_post_order: list[str] = []
        # self.mod_bw_post_order: list[str] = []
        self.total_runtime: float = 0.0

    # Adapted from: https://github.com/pytorch/pytorch/blob/9b902b3ee3bd608a19543362b66bf06c373dd374/torch/_subclasses/fake_tensor.py#L1969  # noqa: PGH004,B950
    # NB: returns fake tensors
    @classmethod
    def _maybe_run_and_benchmark_fallback_kernel(  # type: ignore[no-untyped-def]
        cls,
        func,
        args,
        kwargs,
        orig_not_implemented_exception,
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
            raise orig_not_implemented_exception

        inp_impls = {}
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))
        # Don't use in_kernel_invocation_manager(fake_mode) as we want to do
        # REAL compute (not with meta device)
        with no_dispatch():

            def to_real_tensor(e):  # type: ignore[no-untyped-def]
                if cls.fake_mode.is_our_fake(e):
                    if e.dtype in cls._float_types:
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
            mean_op_time = cuda_time / actual_iters

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
                raise orig_not_implemented_exception

            if isinstance(e, torch.Tensor):
                if id(e) in inp_impls:
                    return inp_impls[id(e)]
                else:
                    return cls.fake_mode.fake_tensor_converter.from_real_tensor(
                        cls.fake_mode, e
                    )
            else:
                return e

        return (pytree.tree_map(map_out, r), mean_op_time)

    def _benchmark_estimate(self, func, args, kwargs) -> tuple[Any, float]:  # type: ignore[no-untyped-def]
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
        assert isinstance(
            self.fake_mode, FakeTensorMode
        ), "Initialize/Assign FakeTensorMode before using this function"
        mean_op_time = 0.0
        if func._overloadpacket not in _VIEW_OPS:
            try:
                res, mean_op_time = self._maybe_run_and_benchmark_fallback_kernel(
                    func,
                    args,
                    kwargs,
                    NotImplementedError,
                )
                return (res, mean_op_time)
            except NotImplementedError:
                RuntimeEstimator._no_fallback_kernel.add(func._overloadpacket)
        res = func(*args, **kwargs or {})
        return (res, mean_op_time)

    def _roofline_estimate(self, func, args, kwargs) -> tuple[Any, float]:  # type: ignore[no-untyped-def]
        def get_num_bytes(t: torch.Tensor) -> int:
            """
            Calculates the memory consumption of a tensor.

            Args:
                t (torch.Tensor): The input tensor.

            Returns:
                int: The memory consumption of the tensor in bytes.
            """
            num_bytes = t.untyped_storage().nbytes()
            mem_consumed = (
                math.ceil(num_bytes / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE
            )
            return mem_consumed

        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        op_time = 0.0
        func_packet = func._overloadpacket
        if func_packet not in _IGNORE_OPS:
            flat_args_kwargs, _ = pytree.tree_flatten((args, kwargs))
            flat_outs, _ = pytree.tree_flatten(out)

            # Get the output dtypes of the operator
            out_dtypes = [
                t.dtype
                for t in flat_outs
                if isinstance(t, torch.Tensor) and t.dtype in self._float_types
            ]
            if not out_dtypes:
                return (out, 0.0)

            if func_packet in flop_registry:
                # Calculate the operator time if it's in flop_registry
                op_count_func = flop_registry[func_packet]
                op_flops, mem_accessed = op_count_func(*args, **kwargs, out_val=out)
                op_bytes = out_dtypes[0].itemsize * mem_accessed

                # Calculate the compute time and memory time from the cluster performance model
                compute_time = self.options.cluster_env.get_device_compute_time(
                    out_dtypes[0], op_flops)
                memory_time = self.options.cluster_env.get_device_memory_time(op_bytes)

                # Return the maximum of compute time and memory time
                op_time = max(compute_time, memory_time)
                # print(f" {func_packet=} {out_dtypes=} {op_flops=} {op_bytes=} {compute_time=} {memory_time=} {op_time=}")
            else:
                # Otherwise, we treat it as a memory-bound operator
                # Estimate the memory access bytes based on the input and output shapes
                read_bytes = sum(
                    get_num_bytes(t)
                    for t in flat_args_kwargs
                    if isinstance(t, torch.Tensor)
                )
                write_bytes = sum(
                    get_num_bytes(t) for t in flat_outs if isinstance(t, torch.Tensor)
                )
                op_bytes = read_bytes + write_bytes
                op_time = self.options.cluster_env.get_device_memory_time(op_bytes)
                # print(f" {func_packet=} {out_dtypes=} {op_bytes=} {op_time=}")

        return (out, op_time)


    def display_modulewise_stats(self, depth: int = 2) -> None:
        """
        Displays module-wise statistics collected by ``RuntimeEstimator``.

        Prints the pre-forward and pre-backward execution orders.
        Displays the module-wise forward and backward runtimes in milliseconds.

        Args:
            depth (int): The maximum depth of module hierarchy to display (default to 2).
        """
        print("Pre-Forward Execution Order: ")
        for mod_fqn in self.mod_fw_pre_order:
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(mod_fqn)
        print("Pre-Backward Execution Order: ")
        for mod_fqn in self.mod_bw_pre_order:
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(mod_fqn)
        for mod_fqn, runtimes in self.mod_runtimes.items():
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(
                f"{mod_fqn} fw: {runtimes.get('fw', 0.0):.3f}ms bw: {runtimes.get('bw', 0.0):.3f}ms"
            )

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):  # type: ignore[no-untyped-def]
        kwargs = kwargs if kwargs else {}

        # Skip ops from non-standard dispatch_sizes_strides_policy such as NJT
        if func in {torch.ops.aten.is_contiguous.default,
                    torch.ops.aten.is_contiguous.memory_format,
                    torch.ops.aten.is_strides_like_format.default,
                    torch.ops.aten.is_non_overlapping_and_dense.default,
                    torch.ops.aten.size.default,
                    torch.ops.aten.sym_size.default,
                    torch.ops.aten.stride.default,
                    torch.ops.aten.sym_stride.default,
                    torch.ops.aten.storage_offset.default,
                    torch.ops.aten.sym_storage_offset.default,
                    torch.ops.aten.numel.default,
                    torch.ops.aten.sym_numel.default,
                    torch.ops.aten.dim.default,
                    torch.ops.prim.layout.default}:

            return NotImplemented

        # If we don't have func in flop_registry, see if it can decompose
        if func not in flop_registry and func is not torch.ops.prim.device.default:
            with self:
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return r

        # TODO: @sanketpurandare: Flatten tensors by desugaring the tensor subclasses
        # TODO: @sanketpurandare: Add logic for incorporating communication time
        res, op_time = self._estimate(func, args, kwargs)
        # for par in self._mod_tracker.parents:
        #     if self._mod_tracker.is_bw:
        #         self.mod_runtimes[par]["bw"] += op_time
        #     else:
        #         self.mod_runtimes[par]["fw"] += op_time
        self.total_runtime += op_time
        # print(f"  {func=} {op_time=} total_runtime={self.total_runtime}")
        return res

    def __call__(self, options: torchcapOptions) -> Self:
        """
        Sets the estimate mode type.

        Currently supported modes:
            - "operator-level-benchmark": Estimates runtime using operator benchmarking.
            - "operator-level-cost-model": Estimates runtime using roofline cost model.

        Args:
            estimate_mode_type (str): The type of estimate mode to use.

        Returns:
            RuntimeEstimator: The runtime estimator instance.

        Raises:
            NotImplementedError: If the estimate mode type is not supported.
        """
        if options.perf_model == "roofline":
            estimate_mode_type = "operator-level-cost-model"
        elif options.perf_model == "benchmark":
            estimate_mode_type = "operator-level-benchmark"
        else:
            raise ValueError(f"Invalid model: {options.perf_model}")

        if estimate_mode_type == "operator-level-benchmark":
            self._estimate = self._benchmark_estimate
        elif estimate_mode_type == "operator-level-cost-model":
            self._estimate = self._roofline_estimate
        else:
            raise NotImplementedError(
                f"estimate_mode_type {estimate_mode_type} not supported"
            )
        self._estimate_mode_type = estimate_mode_type
        self.options = options
        return self

    def __enter__(self) -> Self:
        fake_mode = active_fake_mode()
        assert isinstance(
            fake_mode, FakeTensorMode
        ), "No FakeTensorMode found, designed to used under FakeTensorMode"
        RuntimeEstimator.fake_mode = fake_mode
        # self.total_runtime = 0.0
        # self.mod_runtimes = defaultdict(lambda: defaultdict(lambda: 0.0))
        # self.mod_fw_pre_order.clear()
        # self.mod_bw_pre_order.clear()
        # self.mod_fw_post_order.clear()
        # self.mod_bw_post_order.clear()
        # self._mod_tracker.register_user_hooks(
        #     pre_fw_hook=lambda mod, inp: self.mod_fw_pre_order.append(
        #         self._mod_tracker.get_known_fqn(mod)
        #     ),
        #     pre_bw_hook=lambda mod, g_out: self.mod_bw_pre_order.append(
        #         self._mod_tracker.get_known_fqn(mod)
        #     ),
        #     post_fw_hook=lambda mod, inp, out: self.mod_fw_post_order.append(
        #         self._mod_tracker.get_known_fqn(mod)
        #     ),
        #     post_bw_hook=lambda mod, g_inp: self.mod_bw_post_order.append(
        #         self._mod_tracker.get_known_fqn(mod)
        #     ),
        # )
        # self._mod_tracker.__enter__()
        super().__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        # print(
        #     f"Estimated ({self._estimate_mode_type}) "
        #     f"total_time: {self.total_runtime:.3f} ms"
        # )
        if len(self._no_fallback_kernel) > 0:
            print("no_fallback_kernel: ", list(self._no_fallback_kernel))
        super().__exit__(*args)
        # self._mod_tracker.clear_user_hooks()
        # self._mod_tracker.__exit__()