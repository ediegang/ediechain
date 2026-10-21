"""Storage spec for the polars backend. Skipped whole if polars is not installed."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("polars")

import polars as pl  # noqa: E402

from ediechain.core.storage import Store  # noqa: E402
from ediechain.core.task import task  # noqa: E402


@task
def frame(seed: int) -> pl.DataFrame:
    raise NotImplementedError


@task
def column(seed: int) -> pl.Series:
    raise NotImplementedError


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(root=tmp_path)


def test_a_frame_is_stored_as_parquet(store: Store):
    assert store.artifact_path(frame(seed=1)).suffix == ".parquet"


def test_a_frame_survives_a_round_trip(store: Store):
    node = frame(seed=1)
    store.commit(node, pl.DataFrame({"a": [1, 2]}), {"key": node.key})

    assert store.load(node).equals(pl.DataFrame({"a": [1, 2]}))


def test_a_series_survives_a_round_trip(store: Store):
    node = column(seed=1)
    store.commit(node, pl.Series("a", [1, 2]), {"key": node.key})

    assert store.load(node).equals(pl.Series("a", [1, 2]))
