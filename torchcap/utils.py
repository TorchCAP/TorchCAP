import gc
import logging
import psutil

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def log_dist(message: str, ranks=[]):
    my_rank = dist.get_rank() if dist.is_initialized() else -1
    if my_rank != -1 and len(ranks) > 0 and my_rank not in ranks:
        return
    final_message = "[Rank {}] {}".format(my_rank, message)
    logger.info(final_message)


def see_memory_usage(message, enable=True):
    if not enable:
        return
    if dist.is_initialized() and not dist.get_rank() == 0:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    allocated = f"allocated: {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB"
    max_allcoated = f"max_allocated: {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB"
    reservsed = f"reserved: {round(torch.cuda.memory_reserved() / (1024 * 1024 * 1024),2)} GB"
    max_reservsed = f"max_reserved: {round(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024))} GB"
    print(f"{message} | {allocated} | {max_allcoated} | {reservsed} | {max_reservsed}", flush=True)

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    logger.info(f'CPU Virtual Memory: used = {used_GB} GB, percent = {vm_stats.percent}%')

    # get the peak memory to report correct data, so reset the counter for the next call
    torch.cuda.reset_peak_memory_stats()