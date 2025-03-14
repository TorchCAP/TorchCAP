from typing import List, Callable, cast, Sequence, Union, Optional
from itertools import product
import operator

import torch
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    TupleStrategy,
    PlacementStrategy,
    OutputSpecType,
)
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.fx import Node
from torch._ops import OpOverload
from torch.distributed.tensor._ops._pointwise_ops import (
    pointwise_ops,
    linear_pointwise_ops,
)
import torch.utils._pytree as pytree
from torch._subclasses import FakeTensorMode

from torchcap.sharding import (
    _create_placement_strategy,
)


aten = torch.ops.aten

sharding_propagator = DTensor._op_dispatcher.sharding_propagator

strategy_func_registry: dict[OpOverload, list[PlacementStrategy]] = {}


def enumerate_shardings_for_shape(shape, mesh_ndim: int):
    sharding_choices = [Replicate()] + [Shard(dim) for dim in range(len(shape))]
    # Generate all valid sharding combinations
    return list(product(*[sharding_choices] * mesh_ndim))


def enumerate_shardings_for_inputs(node: Node, mesh: DeviceMesh):
    def extract_shapes_from_args(input_nodes: list[Node]):
        """ Uses pytree to extract tensor shapes from node arguments recursively. """
        def get_shape(arg: Node):
            return arg.meta["val"].shape
        return pytree.tree_map_only(Node, get_shape, input_nodes)

    def gen_shardings(shape):
        if isinstance(shape, torch.Size):
            return enumerate_shardings_for_shape(shape, mesh.ndim)
        else:
            return [shape]

    args_shapes = extract_shapes_from_args(node.all_input_nodes)
    all_shardings = list(product(*[gen_shardings(shape) for shape in args_shapes]))

    return all_shardings


def create_replicated_strategy(node: Node, mesh: DeviceMesh) -> "StrategyGroup":
    def make_input_spec(arg: Node) -> DTensorSpec:
        tensor_meta = TensorMeta(
            shape=arg.meta["val"].shape,
            stride=arg.meta["val"].stride(),
            dtype=arg.meta["val"].dtype,
        )
        return DTensorSpec(
            mesh=mesh,
            placements=(Replicate(),),
            tensor_meta=tensor_meta,
        )

    input_specs = [make_input_spec(n) for n in node.all_input_nodes]

    output_meta = TensorMeta(
        shape=node.meta["val"].shape,
        stride=node.meta["val"].stride(),
        dtype=node.meta["val"].dtype,
    )

    strategies = [
        PlacementStrategy(
            input_specs=input_specs,
            output_specs=DTensorSpec(
                mesh=mesh,
                placements=(Replicate(),),
                tensor_meta=output_meta
            ),
        )
    ]

    group = StrategyGroup(node, strategies, output_meta)
    return group


def extract_tensor_meta(node: Node) -> TensorMeta:
    if isinstance(node.meta["val"], Sequence):
        tensor_metas = []
        for val in node.meta["val"]:
            tensor_metas.append(
                TensorMeta(
                    shape=val.shape,
                    stride=val.stride(),
                    dtype=val.dtype,
                )
            )
        return tensor_metas
    else:
        return TensorMeta(
            shape=node.meta["val"].shape,
            stride=node.meta["val"].stride(),
            dtype=node.meta["val"].dtype,
        )


def _register(op):
    def wrapper(strategy_func = None):
        if strategy_func is None:
            def default_strategy_func(node: Node, mesh: DeviceMesh):
                """Default wrapper for all strategies in ShardingPropagator."""
                assert node.target == op, f"op mismatch: {node.target} != {op}"
                args_schema = pytree.tree_map_only(Node, lambda n: n.meta["strategy"], node.args)
                op_schema = OpSchema(
                    op,
                    args_schema=args_schema,
                    kwargs_schema=cast(dict[str, object], node.kwargs),
                )

                op_strategy: OpStrategy = sharding_propagator.op_strategy_funcs[op](mesh, op_schema)
                if isinstance(op_strategy, TupleStrategy):
                    raise ValueError(f"Op {op} returns a tuple of strategies, which is not supported yet.")

                # Populate tensor meta to all strategies
                out_tensor_meta = extract_tensor_meta(node)
                for output_strategy in op_strategy.strategies:
                    output_strategy.output_specs.tensor_meta = out_tensor_meta

                    # in case where the op does not specify input_specs and output_specs
                    # is a DTensorSpec, we use output_specs as the spec for each DTensor
                    # input arg.
                    if output_strategy.input_specs is None:
                        assert isinstance(output_strategy.output_specs, DTensorSpec)
                        output_strategy.input_specs = tuple(
                            output_strategy.output_specs for _ in op_schema.args_schema)

                strategy_group = StrategyGroup(
                    node,
                    op_strategy.strategies,
                    out_tensor_meta,
                )
                return strategy_group
            strategy_func_registry[op] = default_strategy_func
        else:
            strategy_func_registry[op] = strategy_func
    return wrapper


def register_strategy(op):
    def wrapper(strategy_func = None):
        overloads = [op] if not isinstance(op, list) else op
        for overload in overloads:
            if overload in strategy_func_registry:
                strategy_func_registry.pop(overload)
            _register(overload)(strategy_func)
    return wrapper


# Register all strategies in sharding_propagator
register_strategy(
    list(sharding_propagator.op_strategy_funcs.keys())
)()


class StrategyGroup(OpStrategy):
    def __init__(self, node: Node, strategies: List[PlacementStrategy], output_meta: Union[TensorMeta, Sequence[TensorMeta]]):
        super().__init__(strategies)
        self.node = node

        # Deduplicate strategies
        unique_strategies = {
            str(strategy): strategy for strategy in self.strategies
        }
        self.strategies = list(unique_strategies.values())

        # Assume all outputs have the same output spec
        for strategy in self.strategies:
            if isinstance(strategy.output_specs, Sequence):
                strategy.output_specs = strategy.output_specs[0]

        # Keep the first output meta
        if not isinstance(output_meta, TensorMeta) and isinstance(output_meta, Sequence):
            self.output_meta = output_meta[0]
        else:
            self.output_meta = output_meta

        # Populate tensor meta to all strategies
        for strategy in self.strategies:
            strategy.output_specs.tensor_meta = self.output_meta

    def __len__(self):
        return len(self.strategies)

    def __iter__(self):
        return iter(self.strategies)

    def __str__(self):
        return ", ".join(f'"{str(strategy)}"' for strategy in self.strategies)

    def __repr__(self):
        return f"StrategyGroup({self})"

    @property
    def shape(self):
        if isinstance(self.output_meta, TensorMeta):
            return self.output_meta.shape
        else:
            return self.output_meta[0].shape
    
    @property
    def ndim(self):
        if isinstance(self.output_meta, TensorMeta):
            return len(self.output_meta.shape)
        else:
            return len(self.output_meta[0].shape)

    def append(self, strategy: PlacementStrategy):
        self.strategies.append(strategy)

    @staticmethod
    def from_node(node: Node, mesh: DeviceMesh) -> "StrategyGroup":
        if node.op == "placeholder":
            tensor = node.meta["val"]
            tensor_meta = extract_tensor_meta(node)
            if "param_or_buf" in node.meta:
                # all sharding strategies for parameters and buffers
                strategies = [
                    PlacementStrategy(
                        output_specs=DTensorSpec(
                            mesh=mesh,
                            placements=(sharding[0],),
                            tensor_meta=tensor_meta,
                        ),
                    ) for sharding in enumerate_shardings_for_shape(tensor.shape, mesh.ndim)
                ]
            else:
                # replicate strategy for model inputs
                strategies = [
                    PlacementStrategy(
                        output_specs=DTensorSpec(
                            mesh=mesh,
                            placements=(Replicate(),),
                            tensor_meta=tensor_meta
                        ),
                    )
                ]
            strategy_group = StrategyGroup(node, strategies, tensor_meta)
        elif node.op == "call_function":
            if node.target == operator.getitem:
                input_nodes = node.all_input_nodes
                assert (
                    len(input_nodes) == 1
                ), f"non-compute op only support one input now, found node: {node} with length of inputs: {len(node.args)}"
                arg_strategy: StrategyGroup = input_nodes[0].meta["strategy"]
                strategies = []
                for arg in arg_strategy.strategies:
                    strategy = _create_placement_strategy(
                        node,
                        mesh,
                        placements=arg.output_spec.placements,
                        input_specs=arg.output_spec if isinstance(arg.output_spec, Sequence) else (arg.output_spec,),
                    )
                    strategies.append(strategy)
                strategy_group = StrategyGroup(node, strategies, extract_tensor_meta(node))
            elif node.target in strategy_func_registry:
                strategy_group = strategy_func_registry[node.target](node, mesh)
            else:
                print(f"No sharding strategy registered for {node.target}. Default to replicated strategy.")
                strategy_group = create_replicated_strategy(node, mesh)
        else:
            raise ValueError(f"Unsupported op code: {node.op}")

        return strategy_group


@register_strategy(
    aten._scaled_dot_product_efficient_attention.default,
)
def build_scaled_dot_product_efficient_attention_strategy(
    node: Node, mesh: DeviceMesh
) -> StrategyGroup:
    """
    Modified from torch.distributed.tensor._ops._matrix_ops.scaled_dot_product_efficient_attention_strategy
    The original implementation does not generate correct strategy for Context Parallelism when has_attn_bias is True.
    """
    from torch.distributed.tensor._ops._matrix_ops import expand_to_full_mesh_op_strategy

    # NOTE: currently we only support some simple strategies to support tensor parallelism
    q_input_strategy = node.args[0].meta["strategy"]
    assert isinstance(q_input_strategy, OpStrategy)
    # assuming q/k/v have the same shape

    has_attn_bias = node.args[3] is not None
    compute_log_sumexp = node.args[4]

    single_mesh_dim_strategies = []

    # placement list stores placements of [outputs, inputs]
    # in the spda case, we have 2 valid tensor outputs and 3 or 4 tensor inputs
    # first we can always accept full replication for both inputs and outputs
    all_replicate = [
        Replicate(),
        Replicate(),
        None,
        None,
        Replicate(),
        Replicate(),
        Replicate(),
    ]
    if has_attn_bias:
        all_replicate.append(Replicate())  # attn bias
    single_mesh_dim_strategies.append(all_replicate)

    # Context Parallelism: shards on the sequence dim
    single_mesh_dim_strategies.append(
        [
            Shard(2),  # output
            Shard(2),  # logsumexp
            None,  # philox_seed
            None,  # philox_offset
            Shard(2),  # q
            Shard(2),  # k
            Shard(2),  # v
        ]
    )
    if has_attn_bias:
        single_mesh_dim_strategies[-1].append(Shard(2))

    # second we can accept the sharding pattern of tensor parallelism, which
    # shard on the heads dimension
    qkv_sharding = Shard(1)
    output_sharding = Shard(1)
    if compute_log_sumexp:
        logsumexp_sharding: Placement = Shard(1)
    else:
        # empty logsumexp, replicated
        logsumexp_sharding = Replicate()

    num_heads_dim_sharding = [
        output_sharding,
        logsumexp_sharding,
        None,
        None,
        qkv_sharding,
        qkv_sharding,
        qkv_sharding,
    ]
    if has_attn_bias:
        num_heads_dim_sharding.append(Shard(1))
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    attn_bias_strategy = node.args[3].meta["strategy"] if node.args[3] is not None else None
    op_schema = OpSchema(
        node.target,
        args_schema=(q_input_strategy, q_input_strategy, q_input_strategy, attn_bias_strategy, node.args[4]),
        kwargs_schema=cast(dict[str, object], node.kwargs),
    )
    op_strategy = expand_to_full_mesh_op_strategy(
        mesh,
        op_schema,
        single_mesh_dim_strategies,
        input_index=4,
    )

    strategy_group = StrategyGroup(
        node,
        op_strategy.strategies,
        extract_tensor_meta(node),
    )
    return strategy_group


@register_strategy(aten.native_layer_norm.default)
def build_layer_norm_strategy(node: Node, mesh: DeviceMesh) -> OpStrategy:
    from torch.distributed.tensor._ops._math_ops import layer_norm_strategy
    input_strategy = node.args[0].meta["strategy"]
    num_shards = len(input_strategy)

    normalized_shape = node.args[1]
    # only support replicate for weight and bias for now
    weight_strategy = create_replicated_strategy(node.args[2], mesh)
    # extend to match with the number of input shardings
    weight_strategy.strategies *= num_shards
    bias_strategy = create_replicated_strategy(node.args[3], mesh)
    bias_strategy.strategies *= num_shards
    eps = node.args[4]

    op_schema = OpSchema(
        node.target,
        args_schema=(input_strategy, normalized_shape, weight_strategy, bias_strategy, eps),
        kwargs_schema=cast(dict[str, object], node.kwargs),
    )
    op_strategy = layer_norm_strategy(mesh, op_schema)

    output_meta = TensorMeta(
        shape=node.meta["val"][0].shape,
        stride=node.meta["val"][0].stride(),
        dtype=node.meta["val"][0].dtype,
    )

    strategy_group = StrategyGroup(
        node,
        op_strategy.strategies,
        output_meta,
    )
    return strategy_group


def register_view_op_strategy_map(
    aten_op_overload: torch._ops.OpOverload,
    local_op_name: Callable[..., torch.Tensor],
) -> None:
    from torch.distributed.tensor._ops._view_ops import (
        dim_maps,
        DimMap,
        propagate_shape_and_sharding,
    )

    dim_map: Callable[..., DimMap] = dim_maps[local_op_name]

    @register_strategy(aten_op_overload)
    def build_reshape_strategy(node: Node, mesh: DeviceMesh) -> StrategyGroup:
        args_strategy = pytree.tree_map_only(Node, lambda x: x.meta["strategy"], node.args)
        kwargs_strategy = pytree.tree_map_only(Node, lambda x: x.meta["strategy"], node.kwargs)
        rules = dim_map(*args_strategy, **kwargs_strategy)
        input_strategy = cast(OpStrategy, node.args[0].meta["strategy"])
        global_in_shape = input_strategy.shape
        assert global_in_shape is not None, "Shape required."

        output_strategy = OpStrategy([])
        for input_placement_strategy in input_strategy.strategies:
            input_src_spec = input_placement_strategy.output_spec

            input_tgt_placements, output_placements = propagate_shape_and_sharding(
                input_src_spec.placements,
                tuple(global_in_shape),
                rules,
                mesh.shape,
            )

            # TODO: optimize this. we shouldn't simply blindly replicate
            #       unshardable dims ...
            # FIXME: this can be wrong for situations where we have
            #        [Shard(0), Shard(0)]
            input_tgt_spec = DTensorSpec(
                placements=tuple(input_tgt_placements),
                mesh=input_src_spec.mesh,
                tensor_meta=input_src_spec.tensor_meta,
            )
            # redistribute_costs = [
            #     generate_redistribute_costs(input_strategy, input_tgt_spec)
            # ]
            output_meta = extract_tensor_meta(node)
            output_spec = DTensorSpec(mesh=mesh, placements=tuple(output_placements), tensor_meta=output_meta)
            output_strategy.strategies.append(
                PlacementStrategy(
                    output_specs=output_spec,
                    input_specs=(input_tgt_spec,),
                    # redistribute_cost=redistribute_costs,
                )
            )
        group = StrategyGroup(
            node,
            output_strategy.strategies,
            extract_tensor_meta(node),
        )
        return group


view_ops = {
    aten.squeeze.default: torch.squeeze,
    aten.squeeze.dim: torch.squeeze,
    aten.view.default: Tensor.view,
    aten.reshape.default: torch.reshape,
    aten._unsafe_view.default: Tensor.view,
    aten.unsqueeze.default: torch.unsqueeze,
    aten.expand.default: Tensor.expand,
    aten.permute.default: torch.permute,
    aten.repeat.default: Tensor.repeat,
    aten.transpose.int: torch.transpose,
    aten.view_as_complex.default: torch.view_as_complex,
    aten.view_as_real.default: torch.view_as_real,
}
for op_overload, local_op_name in view_ops.items():
    register_view_op_strategy_map(op_overload, local_op_name)
