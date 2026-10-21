"""Storage spec for `PICKLE`, the escape hatch for a return type no handler covers."""

from __future__ import annotations

from pathlib import Path

import pytest

from ediechain.core.storage import NoHandler, Store, handler_for
from ediechain.core.task import task
from ediechain.handlers.pickle import PICKLE


class Model:
    def __init__(self, weights: list[float]) -> None:
        self.weights = weights


@task(store=PICKLE)
def trained(seed: int) -> Model:
    raise NotImplementedError


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(root=tmp_path)


def test_a_pickled_artifact_is_stored_under_the_handler_extension(store: Store):
    assert store.artifact_path(trained(seed=1)).suffix == ".pkl"


def test_an_object_survives_a_round_trip(store: Store):
    node = trained(seed=1)
    store.commit(node, Model([0.5, 1.5]), {"key": node.key})
    loaded = store.load(node)

    assert isinstance(loaded, Model)
    assert loaded.weights == [0.5, 1.5]


def test_pickle_is_never_the_fallback_for_an_unknown_type():
    """Registered for nothing, so a type with no handler still fails at decoration time."""
    with pytest.raises(NoHandler):
        handler_for(Model)
