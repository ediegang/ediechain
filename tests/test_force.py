"""Spec for forcing a node.

A key covers a task's arguments, not its code (decision 9) and not an external source. So when
you force a node, its key does not change — only its content does. Every artifact below it is
then stale at a key that still looks valid, which is why forcing always cascades downstream.

The runner pulls, so no topological sort is involved: a forced node skips its `exists` check and
everything below it recomputes on the way down. §7.5 is the bug taskchain got here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ediechain import IN_MEMORY, AmbiguousTask, Chain, In, Runner, resolve, task

CALLS: list[str] = []
COUNTER = {"n": 0}


@task
def source(seed: int) -> list[int]:
    CALLS.append("source")
    return [seed, seed + 1]


@task
def middle(values: In[list[int]]) -> int:
    CALLS.append("middle")
    return sum(resolve(values))


@task
def leaf(value: In[int]) -> int:
    CALLS.append("leaf")
    return resolve(value) * 10


@task(store=IN_MEMORY)
def memo(value: In[int]) -> int:
    CALLS.append("memo")
    return resolve(value) + 1


@task
def counted(seed: int) -> int:
    """Deliberately not deterministic — decision 16. Each run gives a different answer."""
    CALLS.append("counted")
    COUNTER["n"] += 1
    return COUNTER["n"]


@task
def echo(value: In[int]) -> int:
    CALLS.append("echo")
    return resolve(value)


@task
def pair(first: In[int], second: In[int]) -> int:
    CALLS.append("pair")
    return resolve(first) + resolve(second)


@task(collective=True)
def spread(seed: int) -> int:
    CALLS.append("spread")
    return seed


class Recorder:
    """A rank that logs its collective calls. Ranks that make different calls hang the job."""

    def __init__(self, writes: bool) -> None:
        self.writes = writes
        self.calls: list[str] = []

    def should_write(self) -> bool:
        return self.writes

    def barrier(self) -> None:
        self.calls.append("barrier")

    def agree(self, local: bool) -> bool:
        self.calls.append("agree")
        return local


@pytest.fixture(autouse=True)
def _reset():
    CALLS.clear()
    COUNTER["n"] = 0


@pytest.fixture
def runner(tmp_path: Path) -> Runner:
    return Runner(root=tmp_path)


@pytest.fixture
def built(runner: Runner) -> Chain[tuple]:
    return Chain.of(leaf(middle(source(seed=1)))).on(runner)


class TestForcedSet:
    def test_forcing_a_node_includes_it(self, built: Chain[tuple]):
        assert built[middle].key in built.force(middle).forced

    def test_forcing_cascades_downstream(self, built: Chain[tuple]):
        forced = built.force(middle).forced

        assert built[leaf].key in forced

    def test_forcing_does_not_reach_upstream(self, built: Chain[tuple]):
        forced = built.force(middle).forced

        assert built[source].key not in forced

    def test_forcing_a_terminal_touches_only_itself(self, built: Chain[tuple]):
        assert built.force(leaf).forced == {built[leaf].key}

    def test_forcing_a_root_touches_everything(self, built: Chain[tuple]):
        assert built.force(source).forced == {node.key for node in built}

    def test_a_node_may_be_forced_directly(self, built: Chain[tuple]):
        node = built[middle]

        assert built.force(node).forced == built.force(middle).forced

    def test_a_pasted_line_forces_its_node(self, built: Chain[tuple]):
        node = built[middle]

        assert built.force(f"middle {node.key[:8]}").forced == built.force(middle).forced

    def test_a_key_alone_forces_its_node(self, built: Chain[tuple]):
        assert built.force(built[middle].key[:8]).forced == built.force(middle).forced

    def test_a_name_forces_its_node(self, built: Chain[tuple]):
        assert built.force("middle").forced == built.force(middle).forced

    def test_an_unknown_string_is_refused(self, built: Chain[tuple]):
        with pytest.raises(KeyError):
            built.force("nonsense")

    def test_text_and_tasks_mix(self, built: Chain[tuple]):
        assert built.force("source", leaf).forced == built.force(source).forced

    def test_several_targets_at_once(self, built: Chain[tuple]):
        assert built.force(source, leaf).forced == built.force(source).forced

    def test_force_composes(self, built: Chain[tuple]):
        assert built.force(middle).force(leaf).forced == built.force(middle).forced

    def test_force_returns_a_new_chain(self, built: Chain[tuple]):
        forced = built.force(middle)

        assert forced is not built
        assert built.forced == frozenset()
        assert forced.nodes == built.nodes

    def test_force_survives_rebinding_a_runner(self, built: Chain[tuple], tmp_path: Path):
        forced = built.force(middle).on(Runner(root=tmp_path))

        assert forced.forced == built.force(middle).forced

    def test_an_ambiguous_task_cannot_be_forced(self, runner: Runner):
        pair = Chain.of(leaf(middle(source(seed=1))), leaf(middle(source(seed=2)))).on(runner)

        with pytest.raises(AmbiguousTask, match="middle"):
            pair.force(middle)

    def test_a_task_outside_the_chain_cannot_be_forced(self, built: Chain[tuple]):
        with pytest.raises(KeyError):
            built.force(counted)


class TestRecompute:
    def test_a_forced_node_recomputes_although_it_is_cached(self, built: Chain[tuple]):
        built.value(leaf)
        CALLS.clear()

        built.force(middle).value(leaf)

        assert "middle" in CALLS

    def test_everything_below_a_forced_node_recomputes(self, built: Chain[tuple]):
        built.value(leaf)
        CALLS.clear()

        built.force(middle).value(leaf)

        assert CALLS == ["middle", "leaf"]

    def test_nothing_above_a_forced_node_recomputes(self, built: Chain[tuple]):
        built.value(leaf)
        CALLS.clear()

        built.force(middle).value(leaf)

        assert "source" not in CALLS

    def test_an_unforced_chain_still_loads(self, built: Chain[tuple]):
        built.value(leaf)
        CALLS.clear()

        built.value(leaf)

        assert CALLS == []

    def test_the_artifact_is_overwritten(self, runner: Runner):
        built = Chain.of(echo(counted(seed=1))).on(runner)
        first = built.value(echo)

        assert built.force(counted).value(echo) != first

    def test_the_key_does_not_change_when_forced(self, built: Chain[tuple]):
        assert built.force(middle)[middle].key == built[middle].key

    def test_a_forced_node_recomputes_once_per_runner(self, built: Chain[tuple]):
        built.value(leaf)
        forced = built.force(middle)
        forced.value(leaf)
        CALLS.clear()

        forced.value(leaf)

        assert CALLS == []

    def test_a_new_runner_forces_again(self, built: Chain[tuple], runner: Runner):
        built.value(leaf)
        forced = built.force(middle)
        forced.value(leaf)
        CALLS.clear()

        forced.on(Runner(root=runner.store.root)).value(leaf)

        assert CALLS == ["middle", "leaf"]

    def test_a_forced_in_memory_node_ignores_the_cached_value(self, runner: Runner):
        built = Chain.of(echo(memo(counted(seed=1)))).on(runner)
        built.value(echo)
        CALLS.clear()

        built.force(memo).value(echo)

        assert "memo" in CALLS

    def test_run_recomputes_a_forced_node_like_value_does(self, built: Chain[tuple]):
        built.value(leaf)
        CALLS.clear()

        built.force(middle).run()

        assert CALLS == ["middle", "leaf"]

    def test_a_forced_in_memory_node_runs_once_for_all_its_consumers(self, runner: Runner):
        shared = memo(middle(source(seed=1)))
        built = Chain.of(pair(echo(shared), leaf(shared))).on(runner)
        built.value(pair)
        CALLS.clear()

        built.force(memo).value(pair)

        assert CALLS.count("memo") == 1


class TestRanks:
    @pytest.mark.parametrize(
        ("build", "target"),
        [(lambda: leaf(middle(source(seed=1))), middle), (lambda: echo(spread(seed=1)), spread)],
        ids=["single-writer", "collective"],
    )
    def test_every_rank_makes_the_same_collective_calls_for_a_forced_chain(
        self, tmp_path: Path, build, target
    ):
        """The second request must load on every rank. A rank that recomputes again calls
        `barrier` while the writer calls `agree`, and the job hangs.
        """
        node = build()
        Chain.of(node).on(Runner(root=tmp_path)).value(node)

        writer, follower = Recorder(writes=True), Recorder(writes=False)
        for coordination in (writer, follower):
            forced = Chain.of(node).on(Runner(root=tmp_path, coordination=coordination)).force(target)
            forced.value(node)
            forced.value(node)

        assert follower.calls == writer.calls


class TestSidecar:
    def test_a_forced_artifact_is_marked(self, built: Chain[tuple], runner: Runner):
        built.value(leaf)
        built.force(middle).value(leaf)

        assert runner.store.load_sidecar(built[middle])["forced"] is True

    def test_an_unforced_artifact_is_not_marked(self, built: Chain[tuple], runner: Runner):
        built.value(leaf)

        assert runner.store.load_sidecar(built[middle])["forced"] is False

    def test_the_mark_is_rewritten_on_a_plain_recompute(self, built: Chain[tuple], tmp_path: Path):
        """A forced artifact that is later recomputed normally must stop claiming it was forced."""
        built.value(leaf)
        built.force(middle).value(leaf)

        elsewhere = Runner(root=tmp_path / "elsewhere")
        fresh = Chain.of(leaf(middle(source(seed=1)))).on(elsewhere)
        fresh.value(leaf)

        assert elsewhere.store.load_sidecar(fresh[middle])["forced"] is False
