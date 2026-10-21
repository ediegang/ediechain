"""End to end: one small experiment, run more than once.

Every other file here tests one piece. This one proves they compose — keys from the encoder, a
chain from `@chain`, artifacts from the store, and the runner's warm-run short-circuit across a
cached → in-memory → cached sandwich. It uses plain types only, so it never skips.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from ediechain import IN_MEMORY, Chain, In, Node, Runner, chain, resolve, task

RUNS: list[str] = []


class Dataset:
    """A type no handler covers, the way `DevelopmentData` is not a frame."""

    def __init__(self, rows: list[int]) -> None:
        self.rows = rows


@task
def raw_rows(count: int, seed: int = 0) -> list[int]:
    RUNS.append("raw_rows")
    generator = random.Random(seed)
    return [generator.randrange(100) for _ in range(count)]


@task(store=IN_MEMORY)
def dataset(rows: In[list[int]], drop_below: int = 0) -> Dataset:
    RUNS.append("dataset")
    return Dataset([row for row in resolve(rows) if row >= drop_below])


@task
def statistics(data: In[Dataset]) -> dict[str, int]:
    RUNS.append("statistics")
    rows = resolve(data).rows
    return {"count": len(rows), "total": sum(rows)}


@task
def report(stats: In[dict[str, int]], label: str = "run") -> str:
    RUNS.append("report")
    return f"{label}: {resolve(stats)['count']} rows"


@chain
def experiment(label: str = "run", count: int = 20, drop_below: int = 0) -> Node[str]:
    return report(statistics(dataset(raw_rows(count=count), drop_below=drop_below)), label=label)


def terminal(label: str = "run", count: int = 20, drop_below: int = 0) -> Node[str]:
    return experiment(label=label, count=count, drop_below=drop_below).outputs


@pytest.fixture(autouse=True)
def _reset_runs():
    RUNS.clear()


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def runner(store_root: Path) -> Runner:
    return Runner(root=store_root)


class TestColdRun:
    def test_every_task_runs(self, runner: Runner):
        runner.value(terminal())

        assert RUNS == ["raw_rows", "dataset", "statistics", "report"]

    def test_the_result_is_what_the_bodies_computed(self, runner: Runner):
        assert runner.value(terminal()) == "run: 20 rows"

    def test_the_persisted_tasks_land_in_the_store(self, runner: Runner):
        runner.value(terminal())

        assert {path.name for path in runner.store.root.iterdir()} == {"raw_rows", "statistics", "report"}

    def test_the_in_memory_task_leaves_nothing_on_disk(self, runner: Runner):
        runner.value(terminal())

        assert not (runner.store.root / "dataset").exists()

    def test_every_artifact_has_a_sidecar(self, runner: Runner):
        built = experiment().on(runner)
        built.run()

        persisted = [node for node in built if node.task.store is not IN_MEMORY]
        assert all(runner.store.sidecar_path(node).exists() for node in persisted)


class TestWarmRun:
    def test_a_second_run_computes_nothing(self, runner: Runner):
        runner.value(terminal())
        RUNS.clear()

        assert runner.value(terminal()) == "run: 20 rows"
        assert RUNS == []

    def test_the_in_memory_task_is_skipped_too(self, runner: Runner):
        """The sandwich: nothing above a cached node runs, including the unpersisted middle."""
        runner.value(terminal())
        RUNS.clear()
        runner.value(terminal())

        assert "dataset" not in RUNS


class TestInvalidation:
    def test_changing_a_downstream_argument_reuses_the_upstream_artifact(self, runner: Runner):
        runner.value(terminal(label="first"))
        RUNS.clear()

        assert runner.value(terminal(label="second")) == "second: 20 rows"
        assert RUNS == ["report"]

    def test_changing_an_upstream_argument_invalidates_everything_below(self, runner: Runner):
        runner.value(terminal(count=20))
        RUNS.clear()
        runner.value(terminal(count=30))

        assert RUNS == ["raw_rows", "dataset", "statistics", "report"]

    def test_a_changed_in_memory_argument_still_invalidates_downstream(self, runner: Runner):
        """An unpersisted task has no artifact, but its key is still in every key below it."""
        runner.value(terminal(drop_below=0))
        RUNS.clear()
        runner.value(terminal(drop_below=50))

        assert "statistics" in RUNS
        assert "raw_rows" not in RUNS


class TestNormalisation:
    def test_a_cold_run_and_a_warm_run_agree_on_the_type(self, runner: Runner, store_root: Path):
        """`statistics` returns `int` counts under a `dict[str, int]` annotation, so nothing is
        widened. Change the annotation to `float` and this is the test that would catch it.
        """
        node = terminal()
        cold = runner.value(node)

        assert Runner(root=store_root).value(node) == cold


class TestSharing:
    def test_two_experiments_share_their_common_prefix(self, runner: Runner):
        """The reason content addressing is worth the trouble — no bookkeeping, no run ids."""
        runner.value(terminal(label="first"))
        RUNS.clear()

        runner.value(terminal(label="second"))
        runner.value(terminal(label="third"))

        assert RUNS == ["report", "report"]

    def test_identical_experiments_built_separately_are_one_node(self):
        assert Chain.of(terminal(), terminal()).nodes == Chain.of(terminal()).nodes


class TestInspection:
    def test_the_plan_is_topological_and_complete(self):
        built = Chain.of(terminal())

        assert [node.task.name for node in built] == ["raw_rows", "dataset", "statistics", "report"]

    def test_printing_the_plan_runs_nothing(self):
        printed = str(Chain.of(terminal()))

        assert RUNS == []
        assert printed.count("\n") == 3

    def test_the_sidecar_matches_the_artifact_it_sits_beside(self, runner: Runner):
        node = terminal()
        runner.value(node)
        info = runner.store.load_sidecar(node)

        assert info["key"] == runner.store.node_path(node).name
        assert info["task"]["name"] == "report"
        assert info["arguments"]["label"] == "run"
