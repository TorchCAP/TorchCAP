import pytest

import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)    

import torchcap


def get_model():
    model_args = ModelArgs()
    model = Transformer(model_args)
    return model, model_args


class TestAPI:

    def test_estimate(self):
        torch.set_default_device("cuda")
        print()

        model, model_args = get_model()

        x = torch.randint(0, model_args.vocab_size, (2, 16))

        graph_info = torchcap.estimate(model, (x,))

