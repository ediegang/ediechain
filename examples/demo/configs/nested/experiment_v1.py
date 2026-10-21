"""A sweep over the clip width, with three terminals instead of one.

`baseline` is `experiment_v0` at its defaults, down to the argument. Run v0 first and this chain
recomputes two branches, not three: the shared nodes are found by key, with no bookkeeping to
tell it they are the same.
"""

from __future__ import annotations

from typing import NamedTuple

from ediechain import Node, chain

from demo.tasks import accuracy, features, predictions, raw_data, trained_model


class Sweep(NamedTuple):
    tight: Node[float]
    baseline: Node[float]
    loose: Node[float]


@chain
def experiment(seed: int = 0) -> Sweep:
    raw = raw_data(seed=seed)
    model = trained_model(seed=seed)
    return Sweep(
        tight=accuracy(predictions(model, features(raw, clip_at=1.0))),
        baseline=accuracy(predictions(model, features(raw, clip_at=2.0))),
        loose=accuracy(predictions(model, features(raw, clip_at=3.0))),
    )
