import logging
import os

import torch.distributed as dist

logger = logging.getLogger()


def init_logger():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] "
        "[%(name)s:%(lineno)d:%(funcName)s] %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


def print_rank(message: str):
    print(f"[Rank {dist.get_rank()}] {message}")


def print_rank_0(message: str):
    if dist.get_rank() == 0:
        print_rank(message)
