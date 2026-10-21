"""The ediechain layer: every step of the pipeline, once.

`demo.data`, `demo.model` and `demo.metrics` know nothing about ediechain and are testable on
their own. This module wraps them as tasks. A config under `configs/` picks the arguments and
composes them into an experiment; it never defines a task, so two experiments that ask for the
same step ask for the same key and share the artifact.
"""

from __future__ import annotations

import polars as pl
from ediechain import IN_MEMORY, In, resolve, task

from demo.data import load_data
from demo.metrics import accuracy as measure_accuracy
from demo.model import RandomModel, random_model


@task
def raw_data(rows: int = 500, seed: int = 0) -> pl.DataFrame:
    return load_data(rows=rows, seed=seed)


@task
def features(data: In[pl.DataFrame], clip_at: float = 2.0) -> pl.DataFrame:
    frame = resolve(data)
    return frame.with_columns(
        pl.col("x1").clip(-clip_at, clip_at),
        pl.col("x2").clip(-clip_at, clip_at),
        (pl.col("x1") * pl.col("x2")).alias("x1x2"),
    )

@task(store=IN_MEMORY, name="model")
def trained_model(seed: int = 0, dimensions: int = 3) -> RandomModel:
    """No handler covers `RandomModel`, so it stays in memory and is rebuilt on every run.

    Its key still reaches everything downstream, because `@converted_by(by_init_params())` on the
    class tells the encoder how to describe it.
    """
    return random_model(seed=seed, dimensions=dimensions)


@task
def predictions(trained: In[RandomModel], data: In[pl.DataFrame]) -> pl.DataFrame:
    frame = resolve(data)
    scores = resolve(trained).score(frame.select("x1", "x2", "x1x2"))
    return frame.select("label").with_columns(scores, (scores > 0).cast(pl.Int64).alias("predicted"))


@task
def accuracy(scored: In[pl.DataFrame]) -> float:
    return measure_accuracy(resolve(scored))
