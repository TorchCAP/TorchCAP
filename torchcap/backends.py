from typing import (
    Any,
    Callable,
    Sequence,
    Tuple,
)

import torch
from torch.fx import GraphModule
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.shape_prop import ShapeProp
from torch._guards import active_fake_mode


class GraphRecorderBackend:

    def __init__(self, is_training: bool):
        self.is_training = is_training
        self.graphs: list[GraphModule] = []
        self.joint_graphs: list[GraphModule] = []
        self.fw_graphs: list[GraphModule] = []
        self.bw_graphs: list[GraphModule] = []

    def __call__(
        self, gm: GraphModule, example_inputs: list[torch.Tensor]
    ) -> Callable[..., Any]:
        fake_mode = active_fake_mode()
        # print(f"{fake_mode}")
        # Create meta["val"] to each node
        FakeTensorProp(gm, fake_mode).propagate(*example_inputs)
        # Create meta["tensor_meta"] to each node
        ShapeProp(gm, fake_mode).propagate(*example_inputs)

        self.graphs.append(gm)

        def fw_compiler(gm: GraphModule, example_inputs: list[torch.Tensor]):
            self.fw_graphs.append(gm)
            return gm.forward

        def bw_compiler(gm: GraphModule, example_inputs: list[torch.Tensor]):
            self.bw_graphs.append(gm)
            return gm.forward

        def partition_fn(
            gm: GraphModule,
            joint_inputs: Sequence[object],
            **kwargs: object,
        ) -> Tuple[GraphModule, GraphModule]:
            self.joint_graphs.append(gm)
            return min_cut_rematerialization_partition(
                gm, joint_inputs, **kwargs
            )

        if self.is_training:
            return aot_autograd(
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
                partition_fn=partition_fn,
            )(gm, example_inputs)
        else:
            return gm.forward
