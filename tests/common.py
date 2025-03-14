import os
import pytest

import torch
import torch.multiprocessing as mp
import torch.distributed as dist


class DistributedTest:
    world_size = 2

    @pytest.fixture(autouse=True)
    def start_processes(self, request):
        """Initialize the distributed environment and run the specific test."""
        selected_test = request.node.name if "test_" in request.node.name else None
        print(f"[DEBUG] start_processes {selected_test}")
        mp.set_start_method('spawn', force=True)
        procs = []
        for i in range(self.world_size):
            p = mp.Process(target=self.run_test, args=(i, self.world_size, request.node.cls, selected_test))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    @staticmethod
    def run_test(rank, world_size, test_cls, selected_test):
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        torch.set_default_device("cuda")
        torch.cuda.set_device(rank)

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

        test_instance = test_cls()

        if selected_test:
            # Run only the selected test
            print(f"[DEBUG] running {selected_test} {dist.is_initialized()}")
            getattr(test_instance, selected_test)()
        else:
            # Run all test methods in the class
            for test_name in dir(test_instance):
                if test_name.startswith("test_"):
                    getattr(test_instance, test_name)()

        dist.destroy_process_group()
