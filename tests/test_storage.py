"""Spec for storage: the `TypeHandler` table and the `Store`.

A handler knows one thing — how to turn a value into bytes at a path and back. It never learns
where the artifact really lives, because the store hands it a temp path and does the rename
itself. Atomicity is then a property of the design, not a rule every handler has to remember.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, NamedTuple

import pytest

from ediechain.core.node import Node
from ediechain.core.storage import JSON, ArtifactExists, NoHandler, Store, handler_for
from ediechain.core.task import In, task


@task
def numbers(seed: int) -> list[int]:
    raise NotImplementedError


@task
def report(raw: In[list[int]]) -> str:
    raise NotImplementedError


@task
def notify(raw: In[str]) -> None:
    raise NotImplementedError


@task
def by_id(seed: int) -> dict[int, str]:
    raise NotImplementedError


@task
def pairs(seed: int) -> list[tuple[int, int]]:
    raise NotImplementedError


@task
def by_name(seed: int) -> dict[str, str]:
    raise NotImplementedError


@task
def widened(seed: int) -> dict[str, float]:
    """A count is an int, and `float` is how most people annotate a dict of statistics."""
    raise NotImplementedError


@task
def scores(seed: int) -> dict[tuple[int, int], float]:
    raise NotImplementedError


@dataclass
class Metrics:
    f1: float
    seen: int


class Window(NamedTuple):
    start: int
    end: int


@task(store=JSON)
def measured(seed: int) -> Metrics:
    raise NotImplementedError


@task(store=JSON)
def windowed(seed: int) -> Window:
    raise NotImplementedError


class Exotic:
    """A return type no handler covers. Module level, because `get_type_hints` resolves a task's
    annotations against its module globals and nothing else.
    """


class Counter:
    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Counter) and other.value == self.value


class UpperHandler:
    """The shape a handler has when its format describes itself: no return annotation in sight."""

    extension = "txt"

    def save(self, value: str, path: Path) -> None:
        path.write_text(value.upper())

    def load(self, path: Path) -> str:
        return path.read_text()


class CounterHandler:
    extension = "count"

    def save(self, value: Counter, path: Path) -> None:
        path.write_text(str(value.value))

    def load(self, path: Path) -> Counter:
        return Counter(int(path.read_text()))


UPPER = UpperHandler()
COUNTER = CounterHandler()


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(root=tmp_path)


def commit(store: Store, node: Node[Any], value: Any, *, overwrite: bool = False) -> None:
    """Most of this spec does not care what the sidecar says, only that it lands with the artifact."""
    store.commit(node, value, {"key": node.key}, overwrite=overwrite)


class TestPaths:
    def test_a_node_gets_a_directory_named_by_its_key(self, store: Store):
        """Decision 19. The directory is the unit that is committed, so it is the thing the key names."""
        node = numbers(seed=1)

        assert store.node_path(node) == store.root / "numbers" / node.key

    def test_the_artifact_sits_inside_that_directory(self, store: Store):
        node = numbers(seed=1)

        assert store.artifact_path(node) == store.node_path(node) / f"data.{JSON.extension}"

    def test_the_sidecar_sits_beside_the_artifact(self, store: Store):
        node = numbers(seed=1)

        assert store.sidecar_path(node) == store.node_path(node) / "run_info.json"

    def test_two_nodes_of_one_task_share_a_directory(self, store: Store):
        assert store.node_path(numbers(seed=1)).parent == store.node_path(numbers(seed=2)).parent

    def test_different_tasks_get_different_directories(self, store: Store):
        assert store.node_path(numbers(seed=1)).parent != store.node_path(report(numbers(seed=1))).parent

    def test_the_extension_comes_from_the_handler(self, store: Store):
        assert store.artifact_path(numbers(seed=1)).suffix == f".{JSON.extension}"


class TestRoundTrip:
    def test_a_value_survives_a_save_and_a_load(self, store: Store):
        node = numbers(seed=1)
        commit(store, node, [1, 2, 3])

        assert store.load(node) == [1, 2, 3]

    @pytest.mark.parametrize(
        ("node", "value"),
        [(by_id(seed=1), {1: "a"}), (pairs(seed=1), [(1, 2)])],
        ids=["int-keys", "tuples"],
    )
    def test_the_return_type_puts_back_what_json_flattens(self, store: Store, node, value):
        """Decision 17 wants a cold run and a warm run to hand a consumer the same value. JSON writes
        an int key as a string and a tuple as a list; the return annotation says how to read it back.
        """
        commit(store, node, value)

        assert store.load(node) == value
        assert type(store.load(node)) is type(value)

    def test_the_artifact_itself_stays_plain_json(self, store: Store):
        """The cast lives in the signature, not in the file, so other tools still read the artifact."""
        node = by_id(seed=1)
        commit(store, node, {1: "a"})

        assert json.loads(store.artifact_path(node).read_text()) == {"1": "a"}

    def test_a_value_that_contradicts_the_return_type_is_refused(self, store: Store):
        """An int key under `dict[str, str]` would come back as the string '0'."""
        node = by_name(seed=1)

        with pytest.raises(TypeError):
            commit(store, node, {0: "cat"})

    def test_a_return_type_json_cannot_carry_is_refused(self, store: Store):
        """A tuple key has no string form to cast back from."""
        node = scores(seed=1)

        with pytest.raises(TypeError):
            commit(store, node, {(1, 2): 0.5})

    def test_a_refused_value_leaves_no_artifact(self, store: Store):
        node = by_name(seed=1)
        with pytest.raises(TypeError):
            commit(store, node, {0: "cat"})

        assert not store.exists(node)

    def test_nothing_exists_before_a_save(self, store: Store):
        assert not store.exists(numbers(seed=1))

    def test_something_exists_after_a_save(self, store: Store):
        node = numbers(seed=1)
        commit(store, node, [1])

        assert store.exists(node)

    def test_a_task_returning_none_is_still_computed(self, store: Store):
        """Existence answers 'is it done', so `None` is a result, not an absence (§7.6)."""
        node = notify(report(numbers(seed=1)))
        commit(store, node, None)

        assert store.exists(node)
        assert store.load(node) is None

    def test_loading_something_that_was_never_saved_raises(self, store: Store):
        with pytest.raises(FileNotFoundError):
            store.load(numbers(seed=1))


class TestAtomicWrites:
    """§7.2. The store owns the stage-and-rename, so no handler can skip it."""

    def test_the_handler_never_receives_the_final_path(self, store: Store, monkeypatch):
        seen: list[Path] = []
        node = numbers(seed=1)
        monkeypatch.setattr(JSON, "save", lambda value, path, returns: seen.append(path) or path.touch())

        commit(store, node, [1])

        assert seen and seen[0] != store.artifact_path(node)

    def test_the_node_directory_does_not_exist_while_the_handler_writes(self, store: Store, monkeypatch):
        """The staging directory must not be the final one. A `mkdir` of the final path would make
        the commit rename land on an empty directory, which POSIX replaces instead of refusing.
        """
        existed: list[bool] = []
        node = numbers(seed=1)
        monkeypatch.setattr(
            JSON, "save", lambda value, path, returns: existed.append(store.node_path(node).exists()) or path.touch()
        )

        commit(store, node, [1])

        assert existed == [False]

    def test_a_handler_that_fails_leaves_no_artifact(self, store: Store, monkeypatch):
        node = numbers(seed=1)

        def half_write(value: Any, path: Path, returns: Any) -> None:
            path.write_text("truncated")
            raise RuntimeError("killed mid-write")

        monkeypatch.setattr(JSON, "save", half_write)

        with pytest.raises(RuntimeError):
            commit(store, node, [1])

        assert not store.exists(node)

    def test_a_successful_commit_leaves_no_staging_directory(self, store: Store):
        node = numbers(seed=1)
        commit(store, node, [1])

        assert [path.name for path in store.node_path(node).parent.iterdir()] == [node.key]

    def test_a_second_commit_is_refused(self, store: Store):
        """Decision 17. The rename refuses in the kernel, so there is no check-then-write window.

        The error is `ENOTEMPTY`, a plain `OSError` — an `except FileExistsError` does not see it.
        """
        node = numbers(seed=1)
        commit(store, node, [1])

        with pytest.raises(ArtifactExists):
            commit(store, node, [2])

        assert store.load(node) == [1]

    def test_overwrite_replaces_the_artifact(self, store: Store):
        """The escape hatch force will use."""
        node = numbers(seed=1)
        commit(store, node, [1])
        commit(store, node, [2], overwrite=True)

        assert store.load(node) == [2]

    def test_a_refused_commit_leaves_no_staging_directory(self, store: Store):
        node = numbers(seed=1)
        commit(store, node, [1])

        with pytest.raises(ArtifactExists):
            commit(store, node, [2])

        assert [path.name for path in store.node_path(node).parent.iterdir()] == [node.key]


class TestCommit:
    """Decision 19. The artifact and its sidecar are one commit, so no reader can see half a node."""

    def test_the_artifact_and_the_sidecar_land_together(self, store: Store):
        node = numbers(seed=1)
        store.commit(node, [1], {"key": node.key, "user": "ada"})

        assert store.load(node) == [1]
        assert store.load_sidecar(node)["user"] == "ada"

    def test_a_committed_node_holds_nothing_else(self, store: Store):
        node = numbers(seed=1)
        commit(store, node, [1])

        assert sorted(path.name for path in store.node_path(node).iterdir()) == ["data.json", "run_info.json"]

    def test_a_failed_commit_leaves_neither(self, store: Store, monkeypatch):
        node = numbers(seed=1)
        monkeypatch.setattr(JSON, "save", lambda value, path, returns: (_ for _ in ()).throw(RuntimeError("no")))

        with pytest.raises(RuntimeError):
            commit(store, node, [1])

        assert not store.node_path(node).exists()

    def test_the_loser_of_a_race_keeps_the_winners_sidecar(self, store: Store):
        """The winner's bytes with the loser's run_info would describe a run that produced nothing."""
        node = numbers(seed=1)
        store.commit(node, [1], {"key": node.key, "user": "winner"})

        with pytest.raises(ArtifactExists):
            store.commit(node, [2], {"key": node.key, "user": "loser"})

        assert store.load(node) == [1]
        assert store.load_sidecar(node)["user"] == "winner"

    def test_a_forced_commit_replaces_both(self, store: Store):
        node = numbers(seed=1)
        store.commit(node, [1], {"key": node.key, "user": "first"})
        store.commit(node, [2], {"key": node.key, "user": "second"}, overwrite=True)

        assert store.load(node) == [2]
        assert store.load_sidecar(node)["user"] == "second"

    def test_a_forced_commit_leaves_no_staging_directory(self, store: Store):
        """The old directory moves out before the new one moves in. It must not be left there."""
        node = numbers(seed=1)
        commit(store, node, [1])
        commit(store, node, [2], overwrite=True)

        assert [path.name for path in store.node_path(node).parent.iterdir()] == [node.key]


class TestClean:
    """A `SIGKILL` cannot be caught, so a staged directory outlives the process that made it."""

    def test_clean_removes_a_staged_directory(self, store: Store):
        node = numbers(seed=1)
        staged = store.node_path(node).parent / "tmp.deadbeef"
        staged.mkdir(parents=True)
        (staged / "data.json").write_text("[1]")

        store.clean()

        assert not staged.exists()

    def test_clean_keeps_committed_nodes(self, store: Store):
        node = numbers(seed=1)
        commit(store, node, [1])

        store.clean()

        assert store.load(node) == [1]

    def test_clean_reports_what_it_removed(self, store: Store):
        """A sweep that says nothing cannot be put in a cron job and trusted."""
        task_dir = store.node_path(numbers(seed=1)).parent
        for token in ("tmp.a1", "tmp.b2"):
            (task_dir / token).mkdir(parents=True)

        assert store.clean() == 2

    def test_clean_on_an_empty_store_removes_nothing(self, store: Store):
        assert store.clean() == 0


class TestNormalisation:
    """Decision 17: `runner.value(node)` returns the content that is in the store. JSON has one
    number type, so an int under a `float` annotation is a legal widening that `convert` accepts —
    which would otherwise make the cold run and the warm run disagree.
    """

    def test_commit_returns_what_a_later_load_will_give_back(self, store: Store):
        node = widened(seed=1)

        stored = store.commit(node, {"count": 20}, {"key": node.key})

        assert stored == store.load(node)
        assert type(stored["count"]) is type(store.load(node)["count"])

    def test_the_widened_value_is_a_float_both_times(self, store: Store):
        node = widened(seed=1)

        stored = store.commit(node, {"count": 20}, {"key": node.key})

        assert isinstance(stored["count"], float)

    def test_a_handler_that_does_not_normalise_hands_the_value_straight_back(self, store: Store):
        """A self-describing format stores what it was given, so there is nothing to put back."""

        @task(store=UPPER)
        def shout(seed: int) -> str:
            raise NotImplementedError

        node = shout(seed=1)

        assert store.commit(node, "hello", {"key": node.key}) == "hello"


class TestStructuredReturns:
    """`store=JSON` is how a dataclass or a NamedTuple opts in. The table still guesses nothing."""

    def test_a_dataclass_survives_a_round_trip(self, store: Store):
        node = measured(seed=1)
        commit(store, node, Metrics(f1=0.91, seen=4000))

        assert store.load(node) == Metrics(f1=0.91, seen=4000)

    def test_a_dataclass_is_written_as_an_object(self, store: Store):
        node = measured(seed=1)
        commit(store, node, Metrics(f1=0.91, seen=4000))

        assert json.loads(store.artifact_path(node).read_text()) == {"f1": 0.91, "seen": 4000}

    def test_a_named_tuple_survives_a_round_trip(self, store: Store):
        node = windowed(seed=1)
        commit(store, node, Window(0, 7))

        assert store.load(node) == Window(0, 7)

    def test_a_named_tuple_is_written_as_an_array(self, store: Store):
        """Positional, so adding a field changes the layout of every artifact already written."""
        node = windowed(seed=1)
        commit(store, node, Window(0, 7))

        assert json.loads(store.artifact_path(node).read_text()) == [0, 7]

    def test_a_dataclass_return_still_needs_an_explicit_store(self):
        """Nothing is guessed: the table covers plain data, `store=` covers the rest."""
        with pytest.raises(NoHandler):

            @task
            def build(seed: int) -> Metrics:
                raise NotImplementedError


class TestHandlerSelection:
    def test_json_covers_the_plain_types(self):
        for returns in (str, int, float, bool, list, dict, type(None)):
            assert handler_for(returns) is JSON

    def test_a_parameterized_generic_finds_its_origin(self):
        assert handler_for(list[int]) is JSON

    def test_a_return_type_without_a_handler_is_refused(self):
        with pytest.raises(NoHandler):
            handler_for(Exotic)

    def test_a_task_whose_return_type_has_no_handler_is_refused_at_decoration_time(self):
        """Import time, not three hours into a run."""
        with pytest.raises(NoHandler):

            @task
            def build(seed: int) -> Exotic:
                raise NotImplementedError


class TestExplicitStore:
    """`store=` is the escape hatch for a type the table does not cover. Nothing is guessed."""

    def test_a_custom_handler_overrides_the_return_type(self, store: Store):
        @task(store=UPPER)
        def shout(seed: int) -> str:
            raise NotImplementedError

        node = shout(seed=1)
        commit(store, node, "hello")

        assert store.artifact_path(node).suffix == ".txt"
        assert store.load(node) == "HELLO"

    def test_a_custom_handler_admits_a_type_the_table_does_not_cover(self, store: Store):
        @task(store=COUNTER)
        def build(seed: int) -> Counter:
            raise NotImplementedError

        node = build(seed=1)
        commit(store, node, Counter(7))

        assert store.load(node) == Counter(7)


class Probe:
    """A typed handler that records what the store handed it."""

    extension = "probe"
    typed: ClassVar[Literal[True]] = True

    def __init__(self) -> None:
        self.seen: list[Any] = []

    def save(self, value: Any, path: Path, returns: Any) -> None:
        self.seen.append(returns)
        path.write_text(repr(value))

    def load(self, path: Path, returns: Any) -> Any:
        self.seen.append(returns)
        return path.read_text()


class TestHandlerProtocols:
    """Two protocols, one flag. A format that describes itself never sees a return annotation."""

    def test_a_typed_handler_is_given_the_return_annotation(self, store: Store):
        probe = Probe()

        @task(store=probe)
        def build(seed: int) -> dict[int, str]:
            raise NotImplementedError

        node = build(seed=1)
        commit(store, node, {1: "a"})
        store.load(node)

        assert probe.seen == [dict[int, str], dict[int, str]]

    def test_a_plain_handler_is_called_with_two_arguments(self, store: Store):
        """`UpperHandler` would raise `TypeError` on a third one, so this pins the dispatch."""

        @task(store=UPPER)
        def shout(seed: int) -> str:
            raise NotImplementedError

        node = shout(seed=1)
        commit(store, node, "hello")

        assert store.load(node) == "HELLO"

    def test_the_flag_is_what_selects_the_call(self, store: Store):
        """Not `isinstance`: a runtime-checkable Protocol matches on method names, not on arity."""
        assert getattr(JSON, "typed", False) is True
        assert getattr(UPPER, "typed", False) is False


class TestLazyBackends:
    def test_importing_the_storage_core_does_not_import_polars(self):
        import subprocess
        import sys

        probe = "import sys, ediechain.core.storage as _; print('polars' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

        assert result.stdout.strip() == "False"
