from typing import cast, Optional
import numpy as np
from numpy.typing import NDArray
from itertools import product
from ortools.math_opt.python import mathopt

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.export import ExportedProgram, ExportGraphSignature
from torch.fx import (
    GraphModule,
    Node,
)
import torch.utils._pytree as pytree
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.distributed.tensor import DeviceMesh

from torchcap.cost_model import comm_model
from torchcap.cost_model.cost_model import GraphInfo
from torchcap.sharding import (
    _populate_tensor_meta,
)
from torchcap.logging_utils import print_rank_0
from torchcap.solver.auto_sharding_strategy import StrategyGroup
from torchcap.cluster_env import MeshTopology


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


def enumerate_input_node_specs(node: Node, mesh):
    """ Converts enumerated sharding options into DTensorSpecs using pytree. """
    def to_dtensor_spec(arg, placement):
        assert isinstance(arg, Node)
        spec = DTensorSpec(
            mesh=mesh,
            placements=placement,
        )
        _populate_tensor_meta(arg, spec)
        return spec

    # Get all shardings for each argument
    all_shardings = enumerate_shardings_for_inputs(node, mesh)

    # Generate DTensorSpec for all sharding options
    input_specs_list = []
    for shardings in all_shardings:
        input_specs_list.append(
            [to_dtensor_spec(arg, shard) for arg, shard in zip(node.all_input_nodes, shardings)]
        )
 
    return input_specs_list


def analyze_graph(
    gm: GraphModule,
    graph_signature: ExportGraphSignature,
    mesh_topo: MeshTopology,
) -> dict[tuple[Node, Node], NDArray]:
    node_to_strategy_group: dict[Node, StrategyGroup] = {}

    # Mark the strategies for input nodes
    input_nodes = gm.graph.find_nodes(op="placeholder")
    num_params_and_buffers = len(graph_signature.inputs_to_parameters) + len(
        graph_signature.inputs_to_buffers
    )
    placeholder_idx = 0
    for node in input_nodes:
        if placeholder_idx < num_params_and_buffers:
            node.meta["param_or_buf"] = True
            placeholder_idx += 1

    # Enumerate strategies for all nodes
    for node in gm.graph.nodes:
        print(f"{node.name}: {node.op}")
        if node.op == "placeholder":
            node.meta["strategy"] = StrategyGroup.from_node(node, mesh_topo.get_device_mesh())
        elif node.op == "output":
            continue
        elif node.op == "call_function":
            node.meta["strategy"] = StrategyGroup.from_node(node, mesh_topo.get_device_mesh())
        else:
            raise RuntimeError(f"Unsupported op code {node.op}")

    for node in gm.graph.nodes:
        if "strategy" in node.meta:
            node_to_strategy_group[node] = node.meta["strategy"]
            # print_rank_0(f"{node.name}: {node.meta['strategy']} ({len(node.meta['strategy'])=})")

    def gen_reshard_matrix(src_specs: list[DTensorSpec], dst_specs: list[DTensorSpec]):
        costs = np.zeros((len(src_specs), len(dst_specs)))
        for i, src_spec in enumerate(src_specs):
            for j, dst_spec in enumerate(dst_specs):
                costs[i, j] = comm_model.reshard_cost(src_spec, dst_spec, mesh_topo)
        return costs

    reshard_matrices = {}
    for node in gm.graph.nodes:
        if node.op == "output":
            continue

        for input_idx, input_node in enumerate(node.all_input_nodes):
            src_strategy_group: StrategyGroup = input_node.meta["strategy"]
            src_specs = [
                strategy.output_spec
                for strategy in src_strategy_group.strategies
            ]

            dst_strategy_group: StrategyGroup = node.meta["strategy"]
            dst_specs = [
                strategy.input_spec(input_idx)
                for strategy in dst_strategy_group.strategies
            ]

            # print(f"{input_node.name} ({len(src_specs)}) -> {node.name} ({len(dst_specs)})")
            reshard_matrices[input_node, node] = gen_reshard_matrix(src_specs, dst_specs)

    # debug
    for nodes, matrix in reshard_matrices.items():
        print(f"{nodes[0].name} -> {nodes[1].name}")
        print(matrix)

    return node_to_strategy_group, reshard_matrices


def solve_auto_sharding(
    gm: GraphModule,
    graph_signature: ExportGraphSignature,
    graph_info: GraphInfo,
    mesh_topo: MeshTopology,
):
    node_strategy_group, reshard_matrices = analyze_graph(gm, graph_signature, mesh_topo)

    model = mathopt.Model("auto_sharding")

    # Decision variables

    s: dict[Node, list[mathopt.Variable]] = {}
    for node in gm.graph.nodes:
        if node in node_strategy_group:
            s[node] = [
                model.add_binary_variable(f"s_{node.name}_{i}") 
                for i in range(len(node_strategy_group[node]))
            ]

    # Constraints

    # Exactly one strategy is selected for each node
    for node, s_vars in s.items():
        model.add_linear_constraint(
            sum(s_vars) == 1,
        )

    def linear_product(x: mathopt.Variable, y: mathopt.Variable):
        z = model.add_binary_variable()
        model.add_linear_constraint(z <= x)
        model.add_linear_constraint(z <= y)
        model.add_linear_constraint(z >= x + y - 1)
        return z

    # Total resharding cost
    for node in s.keys():
        for input_node in node.all_input_nodes:
            if (input_node, node) in reshard_matrices:
                for u, v in zip(s[input_node], s[node]):
                    mat = reshard_matrices[input_node, node]
                    e = linear_product(u, v)


def solve_moe_offload(
    gm: GraphModule,
    graph_signature: ExportGraphSignature,
    graph_info: GraphInfo,
    mesh: DeviceMesh,
):
    pass
