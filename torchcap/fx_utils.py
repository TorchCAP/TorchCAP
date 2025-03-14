import warnings

import torch.fx as fx
import torch
from torch.fx.experimental.symbolic_shapes import hint_int
from torch.fx.experimental.proxy_tensor import py_sym_types
from torch.utils._pytree import tree_flatten


def _tensor_nbytes(numel: int, dtype) -> int:
    return numel * dtype.itemsize


def size_of(node: fx.Node) -> int:
    def object_nbytes(x) -> int:
        if not isinstance(x, torch.Tensor):
            return 0
        return _tensor_nbytes(hint_int(x.numel(), fallback=4096), x.dtype)

    if "val" in node.meta:
        val = node.meta["val"]
        if isinstance(val, py_sym_types):
            return 1
        # NB: The fallback values here are meaningless, maybe we should respect
        # torch._inductor.config.unbacked_symint_fallback (but this is a
        # layering violation)
        elif isinstance(val, (list, tuple)):
            return sum(object_nbytes(n) for n in val)
        elif isinstance(val, dict):
            return sum(object_nbytes(n) for _, n in val.items())
        elif isinstance(val, torch.Tensor):
            return object_nbytes(val)

        raise RuntimeError(f"Unknown metadata type {type(val)} on node {node}")
    if node.op == "get_attr" or node.target is torch.ops.aten._assert_scalar.default:
        return 0
    # raise RuntimeError(
    #     f"Node {node} didn't have `val` metadata; we should always have `val` metadata on the nodes."
    # )
    warnings.warn(f"Node {node} didn't have `val` metadata")
    return 0


def dtypes_of(node: fx.Node) -> set[torch.dtype]:
    if "val" in node.meta:
        val = node.meta["val"]
        flat_vals, _ = tree_flatten(val)
        return {v.dtype for v in flat_vals if isinstance(v, torch.Tensor)}
    raise RuntimeError(f"Node {node} didn't have `val` metadata; we should always have `val` metadata on the nodes.")


def materialize_arg(x):
    if isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.Tensor):
        shape = list(x.meta["val"].shape)

        def realize_symbol(d):
            return hint_int(d, fallback=4096)

        shape = [realize_symbol(s) for s in shape]
        return x.meta["val"].new_empty_strided(
            shape, stride=x.meta["tensor_meta"].stride
        )
    elif isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.SymInt):
        return hint_int(x.meta["val"], fallback=4096)
    elif isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.SymFloat):
        return 1.0
    elif isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.SymBool):
        return True
    else:
        return x
