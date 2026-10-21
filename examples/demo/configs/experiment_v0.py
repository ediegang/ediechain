"""Baseline: one accuracy, from one clip width.

No runner is bound here. The notebook picks the store, so the same experiment can be run against
a scratch directory or the team's shared one without editing this file.
"""

from __future__ import annotations

from ediechain import Node, chain

from demo.tasks import accuracy, features, predictions, raw_data, trained_model


@chain
def experiment(seed: int = 0, clip_at: float = 2.0) -> Node[float]:
    data = features(raw_data(seed=seed), clip_at=clip_at)
    return accuracy(predictions(trained_model(seed=seed), data))
