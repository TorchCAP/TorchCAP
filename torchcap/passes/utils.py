from torch.fx.node import Node
from torch.distributed.tensor._op_schema import PlacementStrategy
from torch.export import ExportGraphSignature
from typing import Any, Callable
from functools import reduce


class ParallelPlan:
    def __init__(self, node_strategies: dict[str, PlacementStrategy] = None):
        self.node_strategies = {} if node_strategies is None else node_strategies

    def __repr__(self):
        strategies_str = ",\n  ".join(f"{k}: {v}" for k, v in self.node_strategies.items())
        return (
            "ParallelPlan(\n"
            f"  {strategies_str}\n"
            ")"
        )

def map_reduce(
    a: Any,
    map_fn: Callable[[Any], Any],
    reduce_fn: Callable[[Any, Any], Any],
    initial: Any
) -> Any:
    """
    Recursively map a function over a nested data structure and reduce the results.
    
    Args:
        a: The input data structure to process
        map_fn: Function to apply to leaf nodes
        reduce_fn: Function to combine results (should be associative)
        initial: Initial value for reduction
        
    Returns:
        The reduced result after mapping over the structure
    """
    if isinstance(a, tuple):
        it = (map_reduce(elem, map_fn, reduce_fn, initial) for elem in a)
        # Support NamedTuple (if it has `_fields`) by repacking into original type
        result = type(a)(*it) if hasattr(a, "_fields") else tuple(it)
        return reduce(reduce_fn, result, initial)
    elif isinstance(a, list):
        result = [map_reduce(elem, map_fn, reduce_fn, initial) for elem in a]
        return reduce(reduce_fn, result, initial)
    elif isinstance(a, dict):
        result = {k: map_reduce(v, map_fn, reduce_fn, initial) for k, v in a.items()}
        return reduce(reduce_fn, result.values(), initial)
    elif isinstance(a, slice):
        result = slice(
            map_reduce(a.start, map_fn, reduce_fn, initial),
            map_reduce(a.stop, map_fn, reduce_fn, initial),
            map_reduce(a.step, map_fn, reduce_fn, initial),
        )
        return reduce(reduce_fn, [result.start, result.stop, result.step], initial)
    else:
        return map_fn(a)