from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any

import pytest

from ediechain.core import registry
from ediechain.core.task import TaskDef
from ediechain.encoders import numpy as _numpy_encoders  # noqa: F401
from ediechain.encoders import polars as _polars_encoders  # noqa: F401

_REGISTRY_TABLES = (registry._CONVERTERS, registry._NAMES, registry._TAGS)


@pytest.fixture(autouse=True)
def isolated_registry():
    """The registry is process-global and refuses duplicates, so a test that registers a type
    would otherwise decide whether a later test can run. Restore it after every test.

    The backends are imported above, before the snapshot is taken. `_LAZY` pops an entry for good,
    so a restore that rolled back past a lazy registration would drop the backend permanently. The
    lazy path itself is covered in a subprocess, where this fixture does not reach.
    """
    snapshots = [dict(table) for table in _REGISTRY_TABLES]
    yield
    for table, snapshot in zip(_REGISTRY_TABLES, snapshots, strict=True):
        table.clear()
        table.update(snapshot)


def taskdef(
    fn: Any,
    *,
    name: str | None = None,
    version: int | None = None,
    ignore: Iterable[str] = (),
    transparent: Iterable[str] = (),
) -> TaskDef:
    sig = inspect.signature(fn)
    return TaskDef(
        name=name or fn.__name__,
        fn=fn,
        signature=sig,
        returns=sig.return_annotation,
        version=version,
        ignore=frozenset(ignore),
        transparent=frozenset(transparent),
    )


def raw_movies(source: str) -> str: raise NotImplementedError


def filtered_movies(raw: str, from_year: int = 1945, min_votes: int = 1000) -> str: raise NotImplementedError


def ensemble(models: list[str], weights: dict[str, float] | None = None) -> str: raise NotImplementedError


@pytest.fixture
def raw() -> TaskDef:
    return taskdef(raw_movies)


@pytest.fixture
def filtered() -> TaskDef:
    return taskdef(filtered_movies)
