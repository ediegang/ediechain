"""Spec for the runner.

The runner pulls. It asks a terminal node for its value; a node already on disk is loaded and
nothing upstream of it is touched. That short-circuit is the point of a cached → in-memory →
cached chain, and a forward sweep over the plan does not give it.
"""

from __future__ import annotations

from itertools import count
from pathlib import Path
from typing import NamedTuple

import pytest

from ediechain.core.chain import Chain
from ediechain.core.runner import Runner, SingleProcessCoordination
from ediechain.core.storage import IN_MEMORY, Store
from ediechain.core.task import In, resolve, task

CALLS: list[str] = []


class Bundle:
    """A return type no handler covers, the way `DevelopmentData` is not a frame."""

    def __init__(self, rows: list[int]) -> None:
        self.rows = rows


@task
def rows(seed: int) -> list[int]:
    CALLS.append("rows")
    return [seed, seed + 1]


@task(store=IN_MEMORY)
def bundled(raw: In[list[int]]) -> Bundle:
    CALLS.append("bundled")
    return Bundle(resolve(raw))


@task
def total(bundle: In[Bundle]) -> int:
    CALLS.append("total")
    return sum(resolve(bundle).rows)


@task
def summed(values: In[list[int]]) -> int:
    CALLS.append("summed")
    return sum(resolve(values))


@task
def doubled(value: In[int]) -> int:
    CALLS.append("doubled")
    return resolve(value) * 2


@task
def combined(parts: list[In[int]], label: str = "x") -> int:
    CALLS.append("combined")
    return sum(resolve(part) for part in parts)


@task(store=IN_MEMORY)
def nothing(value: In[int]) -> None:
    CALLS.append("nothing")
    return


@task(collective=True)
def gathered(seed: int) -> int:
    CALLS.append("gathered")
    return seed + 1


@task
def broken(seed: int) -> list[int]:
    CALLS.append("broken")
    raise RuntimeError("the body failed")


@task
def unwritable(seed: int) -> dict[str, object]:
    return {"value": object()}


DRAWS = count()


@task(collective=True)
def drawn(seed: int) -> int:
    """Not deterministic (decision 16), so two ranks that each keep their own result disagree."""
    return next(DRAWS)


class Scaled(NamedTuple):
    value: In[int]
    factor: int


@task
def scaled(spec: Scaled) -> int:
    return resolve(spec.value) * spec.factor


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS.clear()


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(root=tmp_path)


@pytest.fixture
def runner(store: Store) -> Runner:
    return Runner(root=store.root)


class TestPull:
    def test_a_cold_run_computes_the_whole_graph(self, runner: Runner):
        assert runner.value(doubled(summed(rows(seed=1)))) == 6
        assert CALLS == ["rows", "summed", "doubled"]

    def test_the_value_is_written_to_the_store(self, runner: Runner, store: Store):
        node = doubled(summed(rows(seed=1)))
        runner.value(node)

        assert store.exists(node)

    def test_a_warm_run_loads_instead_of_computing(self, store: Store):
        node = doubled(summed(rows(seed=1)))
        Runner(root=store.root).value(node)
        CALLS.clear()

        assert Runner(root=store.root).value(node) == 6
        assert CALLS == []

    def test_an_upstream_task_is_not_run_when_the_downstream_artifact_exists(self, store: Store):
        """The whole reason to pull: nothing above a cached node is touched, or even imported."""
        node = total(bundled(rows(seed=1)))
        Runner(root=store.root).value(node)
        CALLS.clear()

        assert Runner(root=store.root).value(node) == 3
        assert CALLS == []

    def test_a_shared_input_is_computed_once_per_run(self, runner: Runner):
        source = rows(seed=1)
        runner.value(combined([doubled(summed(source)), doubled(summed(source))]))

        assert CALLS.count("rows") == 1

    def test_inputs_nested_in_containers_are_resolved(self, runner: Runner):
        assert runner.value(combined([doubled(summed(rows(seed=1))), doubled(summed(rows(seed=10)))])) == 48

    def test_a_plain_argument_reaches_the_body_untouched(self, runner: Runner):
        assert runner.value(combined([doubled(summed(rows(seed=1)))], label="other")) == 6

    def test_a_named_tuple_argument_reaches_the_body_as_itself(self, runner: Runner):
        assert runner.value(scaled(Scaled(value=doubled(summed(rows(seed=1))), factor=2))) == 12

    def test_run_materializes_every_terminal(self, runner: Runner, store: Store):
        first, second = doubled(summed(rows(seed=1))), doubled(summed(rows(seed=10)))
        runner.run(Chain.of(first, second), frozenset())

        assert store.exists(first)
        assert store.exists(second)


class TestSubstitution:
    def test_the_body_never_sees_a_node(self, runner: Runner):
        """`resolve` raises on a node, so a body that called it proves substitution happened."""
        assert runner.value(total(bundled(rows(seed=1)))) == 3


class TestInMemory:
    def test_an_in_memory_task_writes_nothing(self, runner: Runner, store: Store):
        node = bundled(rows(seed=1))
        runner.value(node)

        assert not any(store.root.rglob(f"{node.key}*"))

    def test_an_in_memory_value_reaches_its_consumer(self, runner: Runner):
        assert runner.value(total(bundled(rows(seed=1)))) == 3

    def test_an_in_memory_task_may_return_a_type_with_no_handler(self, runner: Runner):
        assert isinstance(runner.value(bundled(rows(seed=1))), Bundle)

    def test_an_in_memory_task_runs_once_within_a_run(self, runner: Runner):
        shared = bundled(rows(seed=1))
        runner.value(combined([total(shared), doubled(total(shared))]))

        assert CALLS.count("bundled") == 1

    def test_an_in_memory_task_runs_again_in_a_new_run(self, store: Store):
        node = bundled(rows(seed=1))
        Runner(root=store.root).value(node)
        CALLS.clear()
        Runner(root=store.root).value(node)

        assert CALLS.count("bundled") == 1

    def test_an_in_memory_task_returning_none_is_not_recomputed(self, runner: Runner):
        """`None` is a result. A `.get()` with no default would rerun this every time it is asked."""
        shared = nothing(doubled(summed(rows(seed=1))))
        runner.value(shared)
        runner.value(shared)

        assert CALLS.count("nothing") == 1


class TestFailure:
    def test_a_failing_body_propagates(self, runner: Runner):
        with pytest.raises(RuntimeError, match="the body failed"):
            runner.value(summed(broken(seed=1)))

    def test_a_failing_body_leaves_no_artifact(self, runner: Runner, store: Store):
        node = broken(seed=1)
        with pytest.raises(RuntimeError):
            runner.value(node)

        assert not store.exists(node)

    def test_a_failing_body_leaves_no_sidecar(self, runner: Runner, store: Store):
        node = broken(seed=1)
        with pytest.raises(RuntimeError):
            runner.value(node)

        assert not store.sidecar_path(node).exists()

    def test_a_failing_save_leaves_no_sidecar(self, runner: Runner, store: Store):
        node = unwritable(seed=1)
        with pytest.raises(TypeError):
            runner.value(node)

        assert not store.sidecar_path(node).exists()


class Follower:
    """A rank that participates in the computation but must not write (`DirDataParallel`)."""

    def __init__(self) -> None:
        self.barriers = 0

    def should_write(self) -> bool:
        return False

    def barrier(self) -> None:
        self.barriers += 1

    def agree(self, local: bool) -> bool:
        return local


class TestCoordination:
    """Only the write is restricted. Which ranks run the *body* is `collective`."""

    def test_single_process_writes(self, store: Store):
        node = doubled(summed(rows(seed=1)))
        Runner(root=store.root, coordination=SingleProcessCoordination()).value(node)

        assert store.exists(node)

    def test_a_follower_does_not_run_a_non_collective_body(self, store: Store):
        """Eight ranks make one LLM call, not eight. The others wait and read what the writer wrote."""
        node = doubled(summed(rows(seed=1)))
        Runner(root=store.root).value(node)
        CALLS.clear()

        assert Runner(root=store.root, coordination=Follower()).value(node) == 6
        assert CALLS == []

    def test_a_follower_runs_a_collective_body(self, store: Store):
        """A collective body is one computation spread across ranks, so every rank must enter it."""
        assert Runner(root=store.root, coordination=Follower()).value(gathered(seed=1)) == 2
        assert CALLS == ["gathered"]

    def test_a_follower_running_a_collective_body_writes_nothing(self, store: Store):
        node = gathered(seed=1)
        Runner(root=store.root, coordination=Follower()).value(node)

        assert not store.exists(node)
        assert not store.sidecar_path(node).exists()

    def test_an_in_memory_task_runs_on_a_follower_too(self, store: Store):
        """It is implicitly collective: there is no artifact for a non-writer to load."""
        source = rows(seed=1)
        Runner(root=store.root).value(source)
        CALLS.clear()

        Runner(root=store.root, coordination=Follower()).value(bundled(source))

        assert CALLS == ["bundled"]

    def test_the_barrier_runs_once_per_node(self, store: Store):
        coordination = Follower()
        Runner(root=store.root, coordination=coordination).value(gathered(seed=1))

        assert coordination.barriers == 1

    def test_a_loaded_node_does_not_barrier(self, store: Store):
        node = doubled(summed(rows(seed=1)))
        Runner(root=store.root).value(node)

        coordination = Follower()
        Runner(root=store.root, coordination=coordination).value(node)

        assert coordination.barriers == 0

    def test_every_rank_continues_with_the_stored_value_of_a_collective_task(self, store: Store):
        node = drawn(seed=1)
        stored = Runner(root=store.root).value(node)

        assert Runner(root=store.root, coordination=LateFollower()).value(node) == stored


class Late(SingleProcessCoordination):
    """A writer whose `exists` check ran just before another writer committed."""

    def agree(self, local: bool) -> bool:
        return False


class LateFollower(Follower):
    def agree(self, local: bool) -> bool:
        return False


class TestRace:
    def test_a_writer_that_loses_the_race_returns_the_stored_value(self, store: Store):
        node = drawn(seed=1)
        stored = Runner(root=store.root).value(node)

        assert Runner(root=store.root, coordination=Late()).value(node) == stored

    def test_a_writer_that_loses_the_race_keeps_the_winners_sidecar(self, store: Store):
        node = drawn(seed=1)
        Runner(root=store.root).value(node)
        winner = store.load_sidecar(node)

        Runner(root=store.root, coordination=Late()).value(node)

        assert store.load_sidecar(node) == winner


class TestSidecar:
    def test_a_sidecar_lands_beside_the_artifact(self, runner: Runner, store: Store):
        node = doubled(summed(rows(seed=1)))
        runner.value(node)

        assert store.sidecar_path(node).parent == store.node_path(node)
        assert store.sidecar_path(node).exists()

    def test_the_sidecar_names_the_task_and_the_key(self, runner: Runner, store: Store):
        node = doubled(summed(rows(seed=1)))
        runner.value(node)
        info = store.load_sidecar(node)

        assert info["task"]["name"] == "doubled"
        assert info["key"] == node.key

    def test_the_sidecar_records_the_input_keys(self, runner: Runner, store: Store):
        source = summed(rows(seed=1))
        node = doubled(source)
        runner.value(node)

        assert store.load_sidecar(node)["inputs"] == {"value": source.key}

    def test_the_sidecar_records_the_plain_arguments(self, runner: Runner, store: Store):
        node = combined([doubled(summed(rows(seed=1)))], label="other")
        runner.value(node)

        assert store.load_sidecar(node)["arguments"]["label"] == "other"

    def test_the_sidecar_records_timing_and_user(self, runner: Runner, store: Store):
        node = doubled(summed(rows(seed=1)))
        runner.value(node)
        info = store.load_sidecar(node)

        assert info["started"] <= info["ended"]
        assert info["duration"] >= 0
        assert info["user"]

    def test_an_in_memory_task_has_no_sidecar(self, runner: Runner, store: Store):
        node = bundled(rows(seed=1))
        runner.value(node)

        assert not store.sidecar_path(node).exists()

    def test_the_sidecar_does_not_record_the_module(self, runner: Runner, store: Store):
        """Decision 15. Recording where the task is defined would re-couple data to file layout."""
        node = doubled(summed(rows(seed=1)))
        runner.value(node)

        assert "module" not in str(store.load_sidecar(node))


class TestKeyMaterial:
    def test_the_sidecar_can_be_checked_against_the_filename(self, runner: Runner, store: Store):
        """Why key material is recorded: a changed key format stops being silent (§7.1)."""
        source = summed(rows(seed=1))
        node = doubled(source)
        runner.value(node)
        info = store.load_sidecar(node)

        assert info["key"] == store.node_path(node).name
        assert set(info["inputs"].values()) == {source.key}


def test_an_ignored_parameter_stays_out_of_the_sidecar(store: Store):
    @task(ignore={"verbose"})
    def quiet(seed: int, verbose: bool = False) -> list[int]:
        return [seed]

    node = quiet(seed=1, verbose=True)
    Runner(root=store.root).value(node)

    assert "verbose" not in store.load_sidecar(node)["arguments"]
