from typing import Callable, Any
from numpy.typing import NDArray
import operator
import numpy as np
import math
import os
from functools import reduce

import torch
from torch.fx import Node, map_arg
from torch.export import ExportedProgram
from torch.distributed.tensor import (
    DTensor,
    Placement,
    Replicate,
)
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpStrategy,
    PlacementStrategy,
)
import torch.utils._pytree as pytree
from torch.distributed.tensor.parallel.style import ParallelStyle

from torchcap.cluster_env import MeshTopology
from torchcap.passes.utils import ParallelPlan
from torchcap.passes.tensor_parallel import (
    _create_placement_strategy,
    _get_input_node_fqn,
    _generate_parameter_and_buffer_placements,
)
from torchcap.solver.sharding_strategy import (
    get_sharding_strategies, enumerate_shardings_for_shape, get_fully_replicated_strategy
)
from torchcap.cost_model import comm_model

from ortools.math_opt.python import mathopt
from ortools.math_opt.python import model_parameters


aten = torch.ops.aten


# This value is hard-coded here:
# https://github.com/pytorch/pytorch/blob/5fba5d83f0703ff8077ab65448a998e9ad6598fd/c10/cuda/CUDACachingAllocator.cpp#L117
_PYTORCH_MIN_ALLOCATE = (
    2**9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)

def get_input_node_specs(
    node: Node, strategy: PlacementStrategy
) -> tuple[DTensorSpec, ...]:
    """
    Get the input specs of a node.
    """
    input_specs_list: list[DTensorSpec] = []
    for input_arg in node.all_input_nodes:
        output_spec = strategy.output_specs
        assert isinstance(output_spec, DTensorSpec)
        input_specs_list.append(output_spec)
    return tuple(input_specs_list)


def analyze_sharding(
    program: ExportedProgram,
    mesh_topo: MeshTopology,
    parallel_strategies: dict[str, ParallelStyle],
) -> dict[Node, OpStrategy]:
    gm = program.graph_module
    # graph_signature = program.graph_signature
    mesh = mesh_topo.device_mesh

    parameter_placements = {}
    if parallel_strategies or len(parallel_strategies) > 0:
        parameter_placements = _generate_parameter_and_buffer_placements(
            program.state_dict.keys(), parallel_strategies
        )

    op_strategies: dict[Node, OpStrategy] = {}

    # Enumerate all possible shardings for input nodes
    num_params_and_buffers = len(program.graph_signature.inputs_to_parameters) + len(
        program.graph_signature.inputs_to_buffers
    )
    placeholder_idx: int = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if placeholder_idx < num_params_and_buffers:
                fqn = _get_input_node_fqn(node.name, program.graph_signature)
                if fqn in parameter_placements:
                    shardings = [(parameter_placements[fqn],)]
                else:
                    shardings = enumerate_shardings_for_shape(
                        node.meta["val"].shape, mesh.ndim)
                op_strategies[node] = OpStrategy([
                    _create_placement_strategy(
                        node, mesh, placements=(sharding[0],))
                    for sharding in shardings
                ])
                placeholder_idx += 1
            else:
                op_strategies[node] = OpStrategy([
                    _create_placement_strategy(
                        node, mesh, placements=(Replicate(),))
                ])

    # Enumerate all strategies for the remaining nodes
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta["op_strategy"] = op_strategies[node]
        elif node.op == "call_function":
            # print(f"node: {node.name}, target: {node.target}")
            if node.target == operator.getitem:
                input_nodes = node.all_input_nodes
                assert (
                    len(input_nodes) == 1
                ), f"non-compute op only support one input now, found node: {node} with length of inputs: {len(node.args)}"
                arg_strategy = op_strategies[input_nodes[0]]
                op_strategies[node] = OpStrategy([
                    _create_placement_strategy(
                        node, mesh,
                        placements=strt.output_spec.placements,
                        input_specs=get_input_node_specs(node, strt),
                    )
                    for strt in arg_strategy.strategies
                ])
                node.meta["op_strategy"] = op_strategies[node]
            else:
                if (
                    node.target
                    not in DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs
                ):
                    op_strategy = get_fully_replicated_strategy(node, mesh)
                else:
                    op_strategy = get_sharding_strategies(node, mesh)
                assert op_strategy is not None
                op_strategies[node] = op_strategy
                node.meta["op_strategy"] = op_strategies[node]
        elif node.op == "output":
            node.meta["op_strategy"] = None
        else:
            raise RuntimeError(f"op code {node.op} not supported")

    # for node in gm.graph.nodes:
    #     if "op_strategy" in node.meta:
    #         print(f"node: {node.name}, op_strategy: {node.meta['op_strategy']}")

    return op_strategies


class CostGraph:
    def __init__(self, gm: torch.fx.GraphModule, mesh_topo: MeshTopology, op_strategies: dict[Node, OpStrategy]):
        self.gm = gm
        self.mesh_topo = mesh_topo
        self.edge_costs: dict[tuple[Node, Node], NDArray[np.float32]] = {}

        def gen_reshard_matrix(src_specs: list[DTensorSpec], dst_specs: list[DTensorSpec]):
            costs = np.zeros((len(src_specs), len(dst_specs)))
            for i, src_spec in enumerate(src_specs):
                for j, dst_spec in enumerate(dst_specs):
                    rc = comm_model.reshard_cost(src_spec, dst_spec, mesh_topo)
                    costs[i, j] = rc if rc != float("inf") else 10000000
            return costs

        for node in gm.graph.nodes:
            if node.op == "output":
                continue

            for input_idx, input_node in enumerate(node.all_input_nodes):
                src_op_strategy = op_strategies[input_node]
                src_specs = [
                    strategy.output_spec
                    for strategy in src_op_strategy.strategies
                ]

                dst_op_strategy = op_strategies[node]
                dst_specs = [
                    strategy.input_spec(input_idx)
                    for strategy in dst_op_strategy.strategies
                ]

                self.edge_costs[input_node, node] = gen_reshard_matrix(src_specs, dst_specs)

                # print(
                #     f"{input_node.name} "
                #     f"({', '.join([str(s.placements[0]) for s in src_specs])}) -> "
                #     f"{node.name} "
                #     f"({', '.join([str(s.placements[0]) for s in dst_specs])}):"
                # )
                # print(self.cost_matrices[input_node, node])

    def __getitem__(self, key: tuple[Node, Node]) -> NDArray[np.float32]:
        return self.edge_costs[key]

    def __repr__(self) -> str:
        return f"CostGraph(\n{self.__str__()}\n)"

    def __str__(self) -> str:
        ret = ""
        for (u, v) in self.edge_costs:
            ret += f"{u.name} -> {v.name}:\n"
            ret += f"{self.edge_costs[(u, v)]}\n"
        return ret


def get_mem_consumed(spec: DTensorSpec):
    mem = math.prod(spec.shape) * spec.tensor_meta.dtype.itemsize
    if spec.mesh.device_type == "cuda":
        return math.ceil((mem) / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE
    return mem


def tree_map_reduce(arg: Any, map_fn: Callable[[Any], Any], reduce_fn: Callable[[Any, Any], Any] = operator.add) -> Any:
    result: Any
    arg = pytree.tree_map(map_fn, arg)
    flat_arg, _ = pytree.tree_flatten(arg)
    result = reduce(reduce_fn, flat_arg)
    return result


def is_fully_replicated(strategy: PlacementStrategy) -> bool:
    ret = True
    if isinstance(strategy.input_specs, DTensorSpec):
        ret = pytree.tree_all(
            lambda spec: all([p == Replicate() for p in spec.placements]), strategy.input_specs)
    ret &= pytree.tree_all(
        lambda spec: all([p == Replicate() for p in spec.placements]), strategy.output_specs)
    return ret


def solve_auto_sharding(
    program: ExportedProgram,
    mesh_topo: MeshTopology,
    max_device_memory: float,
    parallel_strategies: dict[str, Placement],
) -> ParallelPlan:
    op_strategies = analyze_sharding(program, mesh_topo, parallel_strategies)
    cost_graph = CostGraph(program.graph_module, mesh_topo, op_strategies)
    # print(cost_graph)

    gm = program.graph_module
    nodes = list(gm.graph.nodes)

    model = mathopt.Model(name="auto_sharding")

    # Decision variables

    # s[n][i] indicates if strategy i is selected for the node
    s: dict[Node, list[mathopt.Variable]] = {}
    for node in gm.graph.nodes:
        if node in op_strategies:
            s[node] = [
                model.add_binary_variable(name=f"s_{node.name}_{i}") 
                for i in range(len(op_strategies[node].strategies))
            ]

    # e[u, v][i, j] indicates if strategy i is selected for the node u and strategy j is selected for the node v
    e: dict[tuple[Node, Node], dict[tuple[int, int], mathopt.Variable]] = {}
    for v in nodes:
        for u in v.all_input_nodes:
            if (u, v) in cost_graph.edge_costs:
                e[u, v] = {}
                for i in range(len(s[u])):
                    for j in range(len(s[v])):
                        e[u, v][i, j] = model.add_binary_variable(name=f"e_{u.name}_{v.name}_{i}_{j}")

    # Constraints

    # [Constraint] Exactly one sharding is selected for each node
    for node, s_vars in s.items():
        model.add_linear_constraint(
            sum(s_vars) == 1,
        )

    # [Constraint] Linearize the non-linear constraint e[u, v][i, j] = s[u][i] * s[v][j]
    for (u, v), E in e.items():
        for i in range(len(s[u])):
            for j in range(len(s[v])):
                model.add_linear_constraint(E[i, j] <= s[u][i])
                model.add_linear_constraint(E[i, j] <= s[v][j])
                model.add_linear_constraint(E[i, j] >= s[u][i] + s[v][j] - 1)
                # print(f"[DEBUG] {E[i, j]} = {s[u][i]} * {s[v][j]}")

    # [Constraint] Express the total resharding cost
    total_reshard_cost = 0
    for v in nodes:
        for u in v.all_input_nodes:
            if (u, v) in cost_graph.edge_costs:
                R = cost_graph[u, v]
                E = e[u, v]
                for i in range(len(s[u])):
                    for j in range(len(s[v])):
                        total_reshard_cost += R[i, j] * e[u, v][i, j]
                        # print(f"[DEBUG] e: {e[u, v][i, j]} R: {R[i, j]}")

    # Analyze the liveness of the node outputs
    node_to_step = {n: i for i, n in enumerate(nodes)}
    liveness: dict[Node, tuple[int, int | None]] = {}

    def register_liveness(n: Node, user: Node):
        if n not in liveness:
            if n.name in program.graph_signature.inputs_to_parameters:
                liveness[n] = (node_to_step[n], None)
            else:
                liveness[n] = (node_to_step[n], node_to_step[user])

    for user in reversed(nodes):
        map_arg(user.args, lambda n: register_liveness(n, user))
        map_arg(user.kwargs, lambda n: register_liveness(n, user))

    # for node, (start, end) in liveness.items():
    #     print(f"[DEBUG] {node.name}: {start} -> {end}")

    # Express the memory changes at each step
    m_deltas: list[mathopt.LinearExpression] = [0 for _ in range(len(nodes) + 1)]
    for _, (start, end) in liveness.items():
        n = nodes[start]
        if n in op_strategies.keys():
            mem_consumed = mathopt.fast_sum([
                tree_map_reduce(strat.output_specs, get_mem_consumed) * s_var
                for strat, s_var in zip(op_strategies[n].strategies, s[n])
            ])
            m_deltas[start] += mem_consumed
            if end is not None:
                m_deltas[end + 1] -= mem_consumed

    # Express the memory at each step
    m = [model.add_variable(name=f"m_{i}") for i in range(len(nodes) + 2)]
    for i, delta in enumerate(m_deltas):
        model.add_linear_constraint(m[i + 1] == m[i] + delta)
        # print(f"[DEBUG] {i}: {m_deltas[i]}")

    # [Constraint] Peak memory constraint
    # model.add_linear_constraint(m[i] <= M)

    # [Objective] Minimize the total resharding cost
    model.minimize(total_reshard_cost)

    # Warm start solving
    # Prefer the fully replicated strategy for all nodes
    model_params = model_parameters.ModelSolveParameters()
    s_hints = {}
    for node, s_vars in s.items():
        for i in range(len(s_vars)):
            if is_fully_replicated(op_strategies[node].strategies[i]):
                s_hints[s_vars[i]] = 1
    model_params.solution_hints.append(
        model_parameters.SolutionHint(
            variable_values=s_hints
        )
    )
    # for h, v in s_hints.items():
    #     if v == 1:
    #         print(f"[DEBUG] hint: {h} {v}")

    params = mathopt.SolveParameters(enable_output=True)
    result = mathopt.solve(model, mathopt.SolverType.HIGHS, params=params)

    if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
        raise RuntimeError(f"Failed to solve the model ({result.termination})")

    print("Found optimal solution")
    print("Objective value:", result.objective_value())

    m_peak = max([result.variable_values()[m[i]] for i in range(len(m))])
    print(f"Peak memory (GB): {m_peak / 2**30} (max_device_memory: {max_device_memory / 2**30} GB)")

    # for i, delta in enumerate(m_deltas):
    #     print(f"[DEBUG] {i} {delta} = {mathopt.evaluate_expression(delta, result.variable_values())}")

    selected_strategies: dict[str, PlacementStrategy] = {}
    for node, s_vars in s.items():
        s_vals = [round(result.variable_values()[v]) for v in s_vars]
        selected_strategy = op_strategies[node].strategies[s_vals.index(1)]
        selected_strategies[node.name] = selected_strategy
        # print(f"{node.name}: {selected_strategy}")

    parallel_plan = ParallelPlan(
        node_strategies=selected_strategies
    )

    return parallel_plan
