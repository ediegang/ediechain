"""One chain, several ranks, one store.

    uv run torchrun --nproc_per_node 2 examples/distributed.py

Run it twice. The second run loads everything and no body executes on any rank.

Three things are worth watching:

- `raw_rows` is an ordinary task. Only the writing rank runs the body; the others wait at the
  barrier and load what it wrote. Eight ranks make one API call, not eight.
- `embedded` is `collective=True`. Every rank runs the body on its own shard and the gather makes
  all of them end with the same value, which is what lets any one of them write it.
- `accelerator` is an `ignore`d parameter. It reaches the body but never the key, so the same
  artifact is found whether the job ran on two ranks or on twenty.
"""

from __future__ import annotations

import random
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import gather_object

from ediechain import In, Node, Runner, chain, resolve, task
from ediechain.coordinators.accelerate import AccelerateCoordination

DATA_ROOT = Path(__file__).parent / ".artifacts"


@task
def raw_rows(rows: int = 2000, seed: int = 0) -> list[float]:
    generator = random.Random(seed)
    return [generator.gauss(0, 1) for _ in range(rows)]


@task(collective=True, ignore={"accelerator"})
def embedded(values: In[list[float]], accelerator: Accelerator, scale: float = 1.0) -> list[float]:
    """Each rank transforms a contiguous shard, then the gather puts the shards back in rank order.

    The body must leave every rank holding the same value. A rank that kept only its own shard
    would disagree with the artifact the writing rank stored, and the next task would read one
    thing on one rank and another elsewhere.
    """
    rows = resolve(values)
    width = -(-len(rows) // accelerator.num_processes)
    mine = rows[accelerator.process_index * width : (accelerator.process_index + 1) * width]
    return list(gather_object([value * scale for value in mine]))


@task
def total(values: In[list[float]]) -> float:
    return sum(resolve(values))


@chain
def pipeline(accelerator: Accelerator, seed: int = 0, scale: float = 1.0) -> Node[float]:
    return total(embedded(raw_rows(seed=seed), accelerator=accelerator, scale=scale))


def unsupported(accelerator: Accelerator) -> str | None:
    """A collective job cannot start if the ranks cannot meet. Find out now, not mid-chain."""
    try:
        accelerator.wait_for_everyone()
    except NotImplementedError as e:
        return str(e)
    return None


def main() -> None:
    accelerator = Accelerator(cpu=True)

    if (reason := unsupported(accelerator)) is not None:
        accelerator.print(
            f"this machine cannot run a distributed barrier, so the example stops here:\n  {reason}\n\n"
            "On a Mac, `accelerate` calls `torch.distributed.barrier(device_ids=[local_rank])` and\n"
            "torch builds the device from the machine accelerator, which is MPS even when the\n"
            "process group is gloo on CPU. Run this on a Linux host instead."
        )
        return

    runner = Runner(root=DATA_ROOT, coordination=AccelerateCoordination(accelerator))
    built = pipeline(accelerator, seed=7).on(runner)

    answer = built.value(built.outputs)

    # Each rank prints its own answer. They must agree; a mismatch means a collective body left
    # the ranks holding different values.
    print(f"rank {accelerator.process_index}: total={answer:.6f}")
    accelerator.wait_for_everyone()
    accelerator.print(f"\nartifacts under {DATA_ROOT}")
    accelerator.print(f"written by rank {accelerator.process_index} only")


if __name__ == "__main__":
    main()
