import torch
import torch.distributed as dist

from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.config_manager import JobConfig

import torchcap


def main(job_config: JobConfig):
    model_name = job_config.model.name
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]

    model_config.vocab_size = 52000
    model_config.max_seq_len = job_config.training.seq_len

    batch_size = job_config.training.batch_size

    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)
    model.to_empty(device="cuda")

    batch = (
        torch.randint(
            0,
            model_config.vocab_size,
            (batch_size, model_config.max_seq_len),
            device="cuda",
        ),
        torch.randint(
            0,
            model_config.vocab_size,
            (batch_size, model_config.max_seq_len),
            device="cuda",
        ),
    )

    input_ids, labels = batch

    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(), labels.flatten(0, 1)
        )

    class ExportWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.add_module("model", model)
        
        def forward(self, input_ids, labels):
            pred = model(input_ids)
            loss = loss_fn(pred, labels)
            return loss

    graph_info = torchcap.estimate(ExportWrapper(), (input_ids,), {"labels": labels})
    graph_info.print_tabular()


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    if dist.is_initialized():
        dist.destroy_process_group()