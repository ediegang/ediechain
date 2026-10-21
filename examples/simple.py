"""A small pipeline: build a frame, transform it, score it with a random model, measure it.

    uv run python examples/simple.py

Run it twice. The second run computes nothing, because every artifact is addressed by the
arguments that produced it.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl

from ediechain import IN_MEMORY, AmbiguousTask, Chain, In, Node, Runner, Task, chain, resolve, task

DATA_ROOT = Path(__file__).parent / ".artifacts"


class RandomModel:
    """No storage handler covers this, which is why its task is `IN_MEMORY`."""

    def __init__(self, weights: list[float], bias: float) -> None:
        self.weights = weights
        self.bias = bias

    def score(self, frame: pl.DataFrame) -> pl.Series:
        total = pl.lit(self.bias)
        for column, weight in zip(frame.columns, self.weights, strict=False):
            total = total + pl.col(column) * weight
        return frame.select(total.alias("score")).to_series()


@task
def raw_data(rows: int = 500, seed: int = 0) -> pl.DataFrame:
    generator = random.Random(seed)
    return pl.DataFrame(
        {
            "x1": [generator.gauss(0, 1) for _ in range(rows)],
            "x2": [generator.gauss(0, 1) for _ in range(rows)],
            "label": [generator.randrange(2) for _ in range(rows)],
        }
    )


@task
def features(data: In[pl.DataFrame], clip_at: float = 2.0) -> pl.DataFrame:
    frame = resolve(data)
    return frame.with_columns(
        pl.col("x1").clip(-clip_at, clip_at),
        pl.col("x2").clip(-clip_at, clip_at),
        (pl.col("x1") * pl.col("x2")).alias("x1x2"),
    )


@task(store=IN_MEMORY, name="random_model")
def model(seed: int = 0, dimensions: int = 3) -> RandomModel:
    generator = random.Random(seed)
    return RandomModel([generator.gauss(0, 1) for _ in range(dimensions)], generator.gauss(0, 1))


@task
def predictions(trained: In[RandomModel], data: In[pl.DataFrame]) -> pl.DataFrame:
    frame = resolve(data)
    scores = resolve(trained).score(frame.select("x1", "x2", "x1x2"))
    return frame.select("label").with_columns(scores, (scores > 0).cast(pl.Int64).alias("predicted"))


@task
def accuracy(scored: In[pl.DataFrame]) -> float:
    frame = resolve(scored)
    return float((frame["label"] == frame["predicted"]).mean())


@chain(root=DATA_ROOT)
def simple(seed: int = 0, clip_at: float = 2.0) -> Node[float]:
    data = features(raw_data(seed=seed), clip_at=clip_at)
    return accuracy(predictions(model(seed=seed), data))


class Compare(NamedTuple):
    tight: Node[float]
    loose: Node[float]


@chain(root=DATA_ROOT)
def compare(seed: int = 0) -> Compare:
    raw = raw_data(seed=seed)
    trained = model(seed=seed)
    return Compare(
        tight=accuracy(predictions(trained, features(raw, clip_at=1.0))),
        loose=accuracy(predictions(trained, features(raw, clip_at=3.0))),
    )


def show(title: str, built: Chain[Any], target: Task[Any, float]) -> None:
    print(f"  {title:<10} {built.value(target):.3f}")


def main() -> None:
    pipe = simple(seed=7)

    print("the chain, before anything runs:\n")

    print(f"\nartifacts go under {DATA_ROOT}\n")
    show("cold", pipe, accuracy)
    show("warm", pipe, accuracy)
    show("reclip", simple(seed=7, clip_at=1.0), accuracy)
    show("reseed", simple(seed=3), accuracy)

    print("\nask for any task in the chain, typed by its return annotation:")
    print(f"  features   {pipe.value(features).shape}")
    print(f"  outputs    {pipe.value(pipe.outputs):.3f}")

    side_by_side = compare(seed=7)
    print("\none chain, `features` used twice:\n")
    print(side_by_side)

    print("\nasking by task is ambiguous, and it says so:")
    try:
        side_by_side.value(features)
    except AmbiguousTask as exc:
        print(f"  {exc}")

    print("\nasking by name works, because the terminals are named:")
    print(f"  tight      {side_by_side.value(side_by_side.outputs.tight):.3f}")
    print(f"  loose      {side_by_side.value(side_by_side.outputs.loose):.3f}")

    print("\nthe same chain on a different runner:")
    elsewhere = pipe.on(Runner(root=DATA_ROOT))
    print(f"  accuracy   {elsewhere.value(accuracy):.3f}")


if __name__ == "__main__":
    main()
