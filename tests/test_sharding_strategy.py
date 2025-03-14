from typing import Sequence, cast
from itertools import product

import torch
from torch.distributed.device_mesh import (
    init_device_mesh,
)

from torchcap.solver.auto_sharding_strategy import StrategyGroup

aten = torch.ops.aten


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return aten.mm.default(x, y)


def main():
    mesh = init_device_mesh("cuda", (2,))
    print(mesh)

    # x = torch.randn(2, 2, dtype=torch.float32, device="cuda")
    x = torch.randn(4, 4, dtype=torch.float32, device="cuda")
    y = torch.randn(4, 4, dtype=torch.float32, device="cuda")

    program = torch.export.export(Model().cuda(), (x, y))
    program = program.run_decompositions()
    program.graph.print_tabular()

    gm = program.graph_module
    graph_signature = program.graph_signature

    z = program.module()(x, y)

    for node in program.graph.nodes:
        if node.op == "call_function":
            node.meta["strategy"] = StrategyGroup.from_node(node, mesh)

    # input_nodes = gm.graph.find_nodes(op="placeholder")
    # num_params_and_buffers = len(graph_signature.inputs_to_parameters) + len(
    #     graph_signature.inputs_to_buffers
    # )
    # placeholder_idx = 0
    # for node in input_nodes:
    #     if placeholder_idx < num_params_and_buffers:
    #         node.meta["strategy"] = StrategyGroup.from_tensor(node.meta["val"], mesh, strategy="all")
    #         placeholder_idx += 1
    #     else:
    #         node.meta["strategy"] = StrategyGroup.from_tensor(node.meta["val"], mesh, strategy="replicate")

    # for node in gm.graph.nodes:
    #     if node.op == "placeholder":
    #         continue
    #     elif node.op == "output":
    #         continue
    #     elif node.op == "call_function":
    #         node.meta["strategy"] = StrategyVector.from_node(node, mesh)
    #     else:
    #         raise RuntimeError(f"Unsupported op code {node.op}")

    # for node in gm.graph.nodes:
    #     if 'strategy' in node.meta:
    #         print(f"{node.target}: {node.meta['strategy']}")


if __name__ == "__main__":
    main()
