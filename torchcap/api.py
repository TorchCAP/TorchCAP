from typing import (
    Callable,
    Optional,
    Any,
    Union,
)

import torch
from torch.export import ExportedProgram
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.tensor.parallel.style import ParallelStyle

from torchcap import cost_model, solver
from torchcap.cost_model.cost_model import GraphInfo
from torchcap.common import CAPConfig
from torchcap.backends import GraphRecorderBackend
from torchcap.cluster_env import MeshTopology
from torchcap.logging_utils import logger
from torchcap.passes.tensor_parallel import tensor_parallel_transformation


def export(
    model: Callable,
    example_args: tuple[Any, ...],
    example_kwargs: Optional[dict[str, Any]] = None,
) -> ExportedProgram:
    logger.info("Tracing model...")
    try:
        program = torch.export.export_for_training(
            model,
            example_args,
            example_kwargs,
            strict=False,
        )
        # print(program)
        # Reset backend cache to avoid memory leak
        torch._dynamo._reset_guarded_backend_cache()
    except Exception as e:
        raise RuntimeError(f"Error tracing model: {e}")
    return program


def optimize(
    model: Callable,
    example_args: tuple[Any, ...],
    example_kwargs: Optional[dict[str, Any]] = None,
    mesh_topo: Optional[MeshTopology] = None,
    parallel_strategies: Optional[dict[str, ParallelStyle]] = None,
    config: Optional[CAPConfig] = None,
) -> torch.nn.Module:
    assert config is not None
    assert mesh_topo is not None or parallel_strategies is not None, \
        "Either device_mesh or sharding_plan must be provided"
    logger.info("Optimizing model...")
    # for n, p in program.state_dict.items():
    #     print(n, p.shape)
    program = export(model, example_args, example_kwargs)

    parallel_plan = solver.solve_auto_sharding(
        program, mesh_topo, config.cluster_env.get_device_memory_capacity_bytes(), parallel_strategies
    )

    program = tensor_parallel_transformation(
        program, mesh_topo, parallel_plan
    )
    return program.module()


def estimate(
    model_or_program: Union[Callable, ExportedProgram],
    example_args: tuple[Any, ...] = None,
    example_kwargs: Optional[dict[str, Any]] = None,
    is_training: bool = False,
    config: CAPConfig = None,
) -> GraphInfo:
    assert config is not None
    example_args = () if example_args is None else example_args
    example_kwargs = {} if example_kwargs is None else example_kwargs
    if not isinstance(model_or_program, ExportedProgram):
        program = export(model_or_program, example_args, example_kwargs)
    else:
        program = model_or_program
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

    recorder = GraphRecorderBackend(is_training=is_training)
    model = torch.compile(program.module(), fullgraph=True, backend=recorder)

    with fake_mode:
        model(*example_args, **example_kwargs)

    if is_training:
        gm = recorder.joint_graphs[0]
    else:
        gm = recorder.graphs[0]
    # gm = program.graph_module

    print("Graph:")
    gm.graph.print_tabular()

    logger.info("Estimating graph cost...")
    with fake_mode:
        graph_info = cost_model.estimate_graph_cost(model, gm, config)
    return graph_info
