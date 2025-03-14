import torch
from torch.nn import ModuleList
from torch._higher_order_ops.invoke_subgraph import mark_compile_region


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 10)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()

    @mark_compile_region
    def infer(self, x: torch.Tensor):
        return self.mlp(x)

    def forward(self, x: torch.Tensor):
        # for i in range(10):
        #     x = self.mlp[i](x)
        x = self.infer(x)
        return x


def main():
    model = MyModel()

    program = torch.export.export(model, (torch.randn(10),))
    program = program.run_decompositions()
    program.graph_module.print_readable()


if __name__ == "__main__":
    main()
