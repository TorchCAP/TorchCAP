from typing import (
    Callable,
    Optional,
    Any,
    Union,
)

import torch
from torch import fx
from torch.export import ExportedProgram
from torch._subclasses.fake_tensor import FakeTensorMode

from torchcap import cost_model, sharding, solver
from torchcap.cost_model.cost_model import GraphInfo
from torchcap.common import torchcapOptions
from torchcap.backends import GraphRecorderBackend
from torchcap.cluster_env import MeshTopology
from torchcap.logging_utils import logger


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
    sharding_plan: Optional[sharding.ShardingPlan] = None,
    config: Optional[torchcapOptions] = None,
) -> torch.nn.Module:
    assert mesh_topo is not None or sharding_plan is not None, \
        "Either device_mesh or sharding_plan must be provided"
    logger.info("Optimizing model...")
    # for n, p in program.state_dict.items():
    #     print(n, p.shape)
    if sharding_plan is not None:
        program = export(model, example_args, example_kwargs)
        program = sharding.manual_sharding(program, sharding_plan)
    else:
        program = export(model, example_args, example_kwargs).run_decompositions()
        graph_info = estimate(program, example_args, example_kwargs, config=config)
        solver.solve_auto_sharding(
            program.graph_module, program.graph_signature, graph_info, mesh_topo)
    return program.module()


def estimate(
    model_or_program: Union[Callable, ExportedProgram],
    example_args: tuple[Any, ...] = None,
    example_kwargs: Optional[dict[str, Any]] = None,
    is_training: bool = False,
    config: torchcapOptions = None,
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
