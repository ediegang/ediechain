import torch
from accelerate import Accelerator
from accelerate.utils import reduce

from ediechain.core.runner import Coordination


class AccelerateCoordination(Coordination):
    def __init__(self, accelerator: Accelerator, per_node: bool = False) -> None:
        self.accelerator = accelerator
        self.per_node = per_node

    def should_write(self) -> bool:
        return self.accelerator.is_local_main_process if self.per_node else self.accelerator.is_main_process

    def barrier(self) -> None:
        self.accelerator.wait_for_everyone()

    def agree(self, local: bool) -> bool:
        # Every rank must hold the artifact. With one store per node a rank that lacks it would
        # otherwise be told to load a file it does not have.
        if self.accelerator.num_processes == 1:
            return local
        votes = torch.tensor([int(local)], device=self.accelerator.device)
        return int(reduce(votes, reduction="sum").item()) == self.accelerator.num_processes
