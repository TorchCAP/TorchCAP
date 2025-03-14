import os
import pytest

import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)
from torch.distributed.tensor import DeviceMesh
from torch.testing._internal.distributed.fake_pg import FakeStore
import torch.distributed as dist

from torchcap import cost_model
from torchcap.api import _export
from torchcap.sharding import ShardingPlan, Shard, Replicate, manual_sharding
from common import DistributedTest

def get_model():
    model_args = ModelArgs()
    model = Transformer(model_args)
    return model, model_args


class TestSharding(DistributedTest):

    def test_manual_sharding(self):
        if not dist.is_initialized():
            return

        torch.manual_seed(0)

        torch.set_default_device("cuda")
        print()

        mesh = DeviceMesh(device_type="cuda", mesh=list(range(int(os.environ["WORLD_SIZE"]))))
        print(f"mesh: {mesh}")

        sharding_plan = ShardingPlan(mesh)
        sharding_plan.placements = {
            "layers.0.feed_forward.w1.weight": Shard(0),
            "layers.0.feed_forward.w2.weight": Shard(1),
            "layers.1.feed_forward.w1.weight": Shard(0),
            "layers.1.feed_forward.w2.weight": Shard(1),
        }

        model, model_args = get_model()

        x = torch.randint(0, model_args.vocab_size, (2, 16))

        program = _export(model, (x,))

        sharded_program = manual_sharding(program, sharding_plan)

        print(sharded_program.module())

        # graph_info = cost_model.estimate_graph_cost(sharded_program, )
        # graph_info.print_tabular()

        with torch.inference_mode():
            y_sharded = sharded_program.module()(x)

        with torch.inference_mode():
            y_ref = model(x)

        print(f"y_sharded: {y_sharded}")
        print(f"y_ref: {y_ref}")
        assert torch.allclose(y_sharded, y_ref)
