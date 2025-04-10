from typing import Sequence
from itertools import product

import torch
from torch.utils._pytree import tree_map_only
from torch.fx import Node
from torch.distributed.tensor import (
    DTensor,
    DeviceMesh,
    Replicate,
    Shard,
    Partial,
)
from torch.distributed.tensor._op_schema import (
    OpStrategy,
    PlacementStrategy,
    PlacementList,
    Placement,
    OpSchema,
)
from torch.distributed.tensor._dtensor_spec import TensorMeta, DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema
from torch.distributed.tensor._ops.utils import register_op_strategy, RuntimeSchemaInfo
from torch.distributed.tensor._ops._matrix_ops import (
    gen_einsum_strategies,
    infer_broadcast_dims_map,
    map_placements_after_broadcast,
    is_tensor_shardable,
    generate_redistribute_costs,
    expand_to_full_mesh_op_strategy,
)


aten = torch.ops.aten


sharding_propagator = DTensor._op_dispatcher.sharding_propagator


@register_op_strategy([aten.linear.default])
def linear_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    input_shape = op_schema.args_schema[0].shape
    num_input_dims = len(input_shape)

    # placement list stores placements of [output, input, weight, bias]
    single_mesh_dim_strategies: list[PlacementList] = []

    # all replicated
    all_replicate: PlacementList = [Replicate()] * 4
    single_mesh_dim_strategies.append(all_replicate)

    # colwise parallel
    colwise_parallel: PlacementList = [Shard(num_input_dims-1), Replicate(), Shard(0), Shard(0)]
    single_mesh_dim_strategies.append(colwise_parallel)

    # rowwise parallel
    rowwise_parallel: PlacementList = [Partial(), Shard(num_input_dims-1), Shard(1), Replicate()]
    single_mesh_dim_strategies.append(rowwise_parallel)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=1
    )


sharding_propagator.op_strategy_funcs.pop(
    aten._scaled_dot_product_efficient_attention.default, None)

@register_op_strategy(
    aten._scaled_dot_product_efficient_attention.default,
    schema_info=RuntimeSchemaInfo(4),
)
def scaled_dot_product_efficient_attention_strategy(
    mesh: DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    # NOTE: currently we only support some simple strategies to support tensor parallelism
    q_input_strategy = op_schema.args_schema[0]
    assert isinstance(q_input_strategy, OpStrategy)
    # assuming q/k/v have the same shape

    has_attn_bias = op_schema.args_schema[3] is not None
    compute_log_sumexp = op_schema.args_schema[4]

    single_mesh_dim_strategies: list[PlacementList] = []

    # placement list stores placements of [outputs, inputs]
    # in the spda case, we have 2 valid tensor outputs and 3 or 4 tensor inputs
    # first we can always accept full replication for both inputs and outputs
    all_replicate: PlacementList = [
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
        ] + [Replicate()] if has_attn_bias else []
    )

    single_mesh_dim_strategies.append(all_replicate)

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

    return expand_to_full_mesh_op_strategy(
        mesh,
        op_schema,
        single_mesh_dim_strategies,
        input_index=4,
    )


@register_op_strategy(
    [aten.native_layer_norm.default],
    schema_info=RuntimeSchemaInfo(1),
)
def layer_norm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    assert len(op_schema.args_schema) == 5
    new_op_schema = op_schema.args_schema
    num_strategies = len(new_op_schema[0].strategies)

    weight_strategy: OpStrategy = new_op_schema[2]
    weight_strategy.strategies = [
        PlacementStrategy(
            input_specs=weight_strategy.strategies[0].input_specs,
            output_specs=DTensorSpec(
                mesh=mesh,
                placements=(Replicate(),),
                tensor_meta=weight_strategy.strategies[0].output_specs.tensor_meta
            ),
        )
    ] * num_strategies

    bias_strategy: OpStrategy = new_op_schema[3]
    bias_strategy.strategies = [
        PlacementStrategy(
            input_specs=bias_strategy.strategies[0].input_specs,
            output_specs=DTensorSpec(
                mesh=mesh,
                placements=(Replicate(),),
                tensor_meta=bias_strategy.strategies[0].output_specs.tensor_meta
            ),
        )
    ] * num_strategies

    new_schema = OpSchema(
        op=op_schema.op,
        args_schema=new_op_schema,
        kwargs_schema=op_schema.kwargs_schema,
    )

    return torch.distributed.tensor._ops._math_ops.layer_norm_strategy(mesh, new_schema)


def enumerate_shardings_for_shape(shape, mesh_ndim: int):
    sharding_choices = [Replicate()] + [Shard(dim) for dim in range(len(shape))]
    # Generate all valid sharding combinations
    return list(product(*[sharding_choices] * mesh_ndim))


def get_fully_replicated_strategy(node: Node, mesh: DeviceMesh) -> OpStrategy:
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

    op_strategy = OpStrategy([])
    op_strategy.strategies = [
        PlacementStrategy(
            input_specs=input_specs,
            output_specs=DTensorSpec(
                mesh=mesh,
                placements=(Replicate(),),
                tensor_meta=output_meta
            ),
        )
    ]

    return op_strategy


def extract_tensor_meta(node: Node) -> tuple[TensorMeta, ...] | None:
    out = node.meta["val"]
    if isinstance(out, torch.Tensor):
        return (
            TensorMeta(
                shape=out.shape,
                stride=out.stride(),
                dtype=out.dtype,
            ),
        )
    elif isinstance(out, (tuple, list)):
        tensor_meta_list = []
        for o in out:
            if isinstance(o, torch.Tensor):
                tensor_meta_list.append(
                TensorMeta(
                    shape=o.shape,
                    stride=o.stride(),
                        dtype=o.dtype,
                    )
                )
            else:
                tensor_meta_list.append(None)
        return (
            tuple(tensor_meta_list)
            if isinstance(out, tuple)
            else tensor_meta_list
        )
    else:
        return None


def get_sharding_strategies(node: Node, mesh: DeviceMesh) -> OpStrategy:
    if node.op != "call_function":
        return None

    # special case op, we don't need to propagate for local
    # scalar. TODO: figure out a better way to handle this
    if node.op is aten._local_scalar_dense.default:
        return None

    if node.target in sharding_propagator.op_strategy_funcs:
        # generate op strategy for the op.
        # mesh = try_find_mesh_from_args(op_schema.op, op_schema.args_schema)
        # swap the args spec with args strategies
        # args_op_strategy = [spec_to_strategy(i) for i in op_schema.args_schema]
        args_op_strategy = tree_map_only(Node, lambda x: x.meta["op_strategy"], node.args)

        # kwargs_op_strategy = {
        #     k: spec_to_strategy(v) for k, v in op_schema.kwargs_schema.items()
        # }
        # We ignore op strategies in kwargs
        kwargs_op_strategy = node.kwargs

        # construct a new OpSchema on args for strategy based propagation
        strategy_schema: OpSchema = OpSchema(
            op=node.target,
            args_schema=tuple(args_op_strategy),
            kwargs_schema=kwargs_op_strategy,
        )

        output_op_strategy: OpStrategy = sharding_propagator.op_strategy_funcs[node.target](
            mesh, strategy_schema
        )

        # Deduplicate the strategies
        strategy_set = set()
        strategies = []
        for strategy in output_op_strategy.strategies:
            if str(strategy) not in strategy_set:
                strategy_set.add(str(strategy))
                strategies.append(strategy)
        output_op_strategy.strategies = strategies

        out_tensor_meta = extract_tensor_meta(node)

        # Populate tensor meta to all sharding strategies
        for strategy in output_op_strategy.strategies:
            if strategy.output_specs is not None:
                # For ops that return multiple outputs, the outputs should have the same output spec
                if isinstance(strategy.output_specs, Sequence):
                    strategy.output_specs = strategy.output_specs[0]
                strategy.output_specs.tensor_meta = out_tensor_meta[0]

                # In case where the op does not specify input_specs and output_specs
                # is a DTensorSpec, we use output_specs as the spec for each DTensor
                # input arg.
                if strategy.input_specs is None:
                    assert isinstance(strategy.output_specs, DTensorSpec)
                    strategy.input_specs = (strategy.output_specs,) \
                        if isinstance(strategy.output_specs, DTensorSpec) \
                            else strategy.output_specs

        return output_op_strategy
