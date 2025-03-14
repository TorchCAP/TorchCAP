import argparse
import os
import gc

import torch
from transformers import AutoTokenizer, AutoModel
from torch.profiler import profile, ProfilerActivity

import torchcap
from torchcap.cost_model.cost_model import round_memory, round_time
from torchcap.cluster_env import ClusterEnv
from torchcap.utils import see_memory_usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-2.7b")
    parser.add_argument("--cluster-env", type=str, default=None)
    args = parser.parse_args()

    torch.cuda.set_device(int(os.environ["RANK"]))
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_bos_token = False

    model: torch.nn.Module = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16)
    model.cuda()

    see_memory_usage("After model creation")

    input_ids = torch.randint(0, 10000, (16, 1024)).cuda()

    # mesh = DeviceMesh(device_type="cuda", mesh=list(range(int(os.environ["WORLD_SIZE"]))))
    # sharding_plan = ShardingPlan(mesh, {})
    # sharding_plan.placements.update({
    #     pname: Shard(0) for pname, _ in model.named_parameters() if "fc1.weight" in pname
    # })
    # sharding_plan.placements.update({
    #     pname: Shard(1) for pname, _ in model.named_parameters() if "fc2.weight" in pname
    # })
    # print(f"Device mesh: {mesh}")
    # print(f"Sharding plan: {sharding_plan}")

    config = torchcap.torchcapOptions(perf_model="roofline")
    if args.cluster_env is not None:
        config.cluster_env = ClusterEnv.from_json(args.cluster_env)
    mesh_topo = config.cluster_env.mesh_topo
    # mesh_topo.mesh = mesh
    mesh_topo.build_device_mesh(mesh=list(range(int(os.environ["WORLD_SIZE"]))))
    print(f"Device mesh: {mesh_topo.get_device_mesh()}")
    model = torchcap.optimize(model, (input_ids,), mesh_topo=mesh_topo, config=config)
    print(model)

    torch.cuda.empty_cache()
    gc.collect()

    see_memory_usage("After optimization")

    graph_info = torchcap.estimate(model, (input_ids,), options=config)
    graph_info.print_tabular()

    see_memory_usage("After estimation")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_max_memory_allocated()

    activities = [ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        niters = 5
        start.record()
        with torch.no_grad():
            for _ in range(niters):
                prof.step()
                model(input_ids)
        end.record()
        torch.cuda.synchronize()
        wall_time_us = start.elapsed_time(end) / niters * 1e3

    events = prof.key_averages()

    t_profile = events.total_average().self_device_time_total / niters
    t_torchcap = graph_info.get_total_runtime()
    t_error = abs(t_profile - t_torchcap) / t_torchcap * 100

    print(f"Wall clock time: {round_time(wall_time_us)}")
    print(f"CUDA runtime (profile): {round_time(t_profile)}")
    print(f"Estimated CUDA runtime (torchcap): {round_time(t_torchcap)}")
    print(f"Error: {t_error:.2f}%")

    m_profile = torch.cuda.max_memory_allocated(torch.cuda.current_device())
    m_torchcap = graph_info.peak_memory
    m_error = abs(m_profile - m_torchcap) / m_torchcap * 100

    print(f"Peak memory (profile): {round_memory(m_profile)}")
    print(f"Estimated peak memory (torchcap): {round_memory(m_torchcap)}")
    print(f"Error: {m_error:.2f}%")


if __name__ == "__main__":
    main()
