"""Storage spec for the numpy backend. Skipped whole if numpy is not installed."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

import numpy as np  # noqa: E402

from ediechain.core.storage import Store  # noqa: E402
from ediechain.core.task import task  # noqa: E402


@task
def array(seed: int) -> np.ndarray:
    raise NotImplementedError


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(root=tmp_path)


def test_the_backend_registers_itself_on_the_first_miss():
    """The lazy path, in a fresh process: `@task` resolves the handler while decorating, so a
    backend missing from the table fails at import time, not at save time.
    """
    probe = (
        "import numpy as np\n"
        "from ediechain import task\n"
        "@task\n"
        "def f() -> np.ndarray: ...\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)

    assert result.stdout.strip() == "ok", result.stderr


def test_an_array_is_stored_under_the_handler_extension(store: Store):
    assert store.artifact_path(array(seed=1)).suffix == ".npy"


def test_an_array_survives_a_round_trip(store: Store):
    node = array(seed=1)
    store.commit(node, np.arange(3), {"key": node.key})
    loaded = store.load(node)

    assert isinstance(loaded, np.ndarray)
    assert np.array_equal(loaded, np.arange(3))


def test_a_two_dimensional_array_keeps_its_shape_and_dtype(store: Store):
    node = array(seed=2)
    value = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    store.commit(node, value, {"key": node.key})
    loaded = store.load(node)

    assert loaded.shape == value.shape
    assert loaded.dtype == value.dtype
