import argparse
import gc
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from torch.profiler import profile, record_function, ProfilerActivity
from torch.autograd import DeviceType
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from torch._subclasses import FakeTensorMode
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch._functorch.aot_autograd import aot_export_module
import torch.utils._pytree as pytree

import torchcap
from torchcap.cost_model.cost_model import round_memory, round_time
from torchcap.cluster_env import ClusterEnv
from torchcap.utils import see_memory_usage
from torchcap.logging_utils import init_logger, logger


class HuggingfaceModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return (outputs[0],)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-2.7b")
    parser.add_argument("--cluster-env", type=str, default=None)
    args = parser.parse_args()

    init_logger()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_bos_token = False

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        return_dict=False,
    )
    model = HuggingfaceModelWrapper(model)
    model.cuda()
    print(model)

    see_memory_usage("After loading model", enable=True)

    input_ids = torch.randint(0, 10000, (16, 1024)).cuda()
    labels = torch.randint(0, 10000, (16, 1024)).cuda()
    input_kwargs = {"input_ids": input_ids, "labels": labels}

    model = torchcap.export(model, example_args=(), example_kwargs=input_kwargs).module()

    options = torchcap.CAPConfig()
    options.perf_model = "roofline"
    if args.cluster_env is not None:
        options.cluster_env = ClusterEnv.from_json(args.cluster_env)
    graph_info = torchcap.estimate(model, example_kwargs=input_kwargs, is_training=False, config=options)
    graph_info.print_tabular()

    torch.cuda.empty_cache()
    gc.collect()

    see_memory_usage("After estimation", enable=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    activities = [ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('/workspace/torchcap/trace/'),
    ) as prof:
        niters = 1
        start.record()
        with torch.no_grad():
            for _ in range(niters):
                prof.step()
                model(**input_kwargs)
        end.record()
        torch.cuda.synchronize()
        wall_time_us = start.elapsed_time(end) / niters * 1e3

    events = prof.key_averages()
    # print(events.table(sort_by="self_device_time_total", row_limit=-1))

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
