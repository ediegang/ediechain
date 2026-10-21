"""Spec for `Chain` and the `@chain` decorator.

`@chain` turns a function that composes tasks into one that returns a `Chain`. Building a chain
walks back over `node.inputs` from whatever the function produced; it runs nothing, touches no
path and reads no file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, assert_type

import pytest

from ediechain import (
    AmbiguousTask,
    Chain,
    DuplicateTaskName,
    In,
    Node,
    NoRunner,
    Runner,
    chain,
    resolve,
    task,
)


@task
def source(path: str) -> str:
    raise NotImplementedError


@task
def clean(raw: In[str], drop_empty: bool = True) -> str:
    raise NotImplementedError


@task
def merge(left: In[str], right: In[str]) -> str:
    raise NotImplementedError


@task
def combine(parts: list[In[str]], weights: dict[str, In[str]] | None = None) -> str:
    raise NotImplementedError


def assert_topological(built: Chain[Any]) -> None:
    position = {node.key: index for index, node in enumerate(built)}
    for node in built:
        for dependency in node.inputs:
            assert position[dependency.key] < position[node.key], f"{dependency} runs after {node}"


class TestContents:
    def test_a_single_node_is_its_own_chain(self):
        node = source("a.csv")

        assert tuple(Chain.of(node)) == (node,)

    def test_a_linear_graph_is_walked_back_to_the_root(self):
        root = source("a.csv")
        middle = clean(root)
        terminal = clean(middle)

        assert tuple(Chain.of(terminal)) == (root, middle, terminal)

    def test_a_shared_input_appears_once(self):
        root = source("a.csv")
        built = Chain.of(merge(clean(root), clean(root, drop_empty=False)))

        assert len(built) == 4
        assert [node.key for node in built].count(root.key) == 1

    def test_two_terminals_share_a_subgraph(self):
        root = source("a.csv")

        assert len(Chain.of(clean(root), clean(root, drop_empty=False))) == 3

    def test_the_same_terminal_passed_twice_is_deduped(self):
        node = source("a.csv")

        assert len(Chain.of(node, node)) == 1

    def test_equal_nodes_built_separately_collapse(self):
        assert len(Chain.of(clean(source("a.csv")), clean(source("a.csv")))) == 2

    def test_a_terminal_that_is_also_an_input_appears_once(self):
        root = source("a.csv")
        terminal = clean(root)

        assert tuple(Chain.of(root, terminal)) == (root, terminal)

    def test_nodes_nested_in_containers_are_found(self):
        left, right = source("a.csv"), source("b.csv")
        built = Chain.of(combine([left], weights={"right": right}))

        assert len(built) == 3
        assert set(built) == {left, right, built.terminals[0]}

    def test_an_empty_chain_is_empty(self):
        built = Chain.of()

        assert len(built) == 0
        assert tuple(built) == ()


class TestOrder:
    def test_every_input_comes_before_its_node(self):
        root = source("a.csv")
        left, right = clean(root), clean(root, drop_empty=False)

        assert_topological(Chain.of(combine([merge(left, right), left, root])))

    def test_a_terminal_comes_last(self):
        terminal = clean(clean(source("a.csv")))

        assert tuple(Chain.of(terminal))[-1] == terminal

    def test_the_order_is_deterministic(self):
        def build() -> Chain[Any]:
            return Chain.of(merge(clean(source("a.csv")), clean(source("b.csv"))))

        assert [node.key for node in build()] == [node.key for node in build()]


class TestTaskNames:
    def test_two_tasks_sharing_a_name_in_one_chain_raise(self):
        @task(name="clash")
        def one(path: str) -> str:
            raise NotImplementedError

        @task(name="clash")
        def two(path: str) -> str:
            raise NotImplementedError

        with pytest.raises(DuplicateTaskName, match="clash"):
            Chain.of(one("a.csv"), two("b.csv"))

    def test_one_task_used_twice_is_fine(self):
        assert len(Chain.of(source("a.csv"), source("b.csv"))) == 2

    def test_a_name_clash_outside_the_chain_is_not_reported(self):
        @task(name="clash")
        def one(path: str) -> str:
            raise NotImplementedError

        @task(name="clash")
        def two(path: str) -> str:
            raise NotImplementedError

        assert len(Chain.of(one("a.csv"))) == 1
        assert len(Chain.of(two("b.csv"))) == 1


class TestLookup:
    def test_a_task_with_one_node_is_found_by_the_task(self):
        root = source("a.csv")
        built = Chain.of(clean(root))

        assert built[source] == root

    def test_a_task_with_several_nodes_is_ambiguous(self):
        """A chain is nodes, not tasks. One task, two argument sets, two nodes."""
        root = source("a.csv")
        built = Chain.of(merge(clean(root), clean(root, drop_empty=False)))

        with pytest.raises(AmbiguousTask, match="clean"):
            built[clean]

    def test_a_task_outside_the_chain_raises(self):
        with pytest.raises(KeyError):
            Chain.of(source("a.csv"))[clean]

    def test_lookup_is_by_identity_not_by_name(self):
        @task(name="source")
        def impostor(path: str) -> str:
            raise NotImplementedError

        with pytest.raises(KeyError):
            Chain.of(source("a.csv"))[impostor]


class TestPastedLookup:
    """A string key, so a line copied out of a rendered view addresses its node.

    A node answers a token when the token equals its task name or is a prefix of its key. A token
    no node answers is noise and is dropped, so a mark and a branch come along for free and the
    core needs to know no glyph: every mark is non-hex and is not a task name. The tokens that
    remain narrow together, so a name and a key pasted from one line select one node.

    It is the untyped lookup. `built[clean]` is `Node[str]`, `built["clean"]` is `Node[Any]`, which
    is the price of addressing a node by text rather than by the task object.
    """

    def test_a_task_name_finds_its_node(self):
        root = source("a.csv")

        assert Chain.of(clean(root))["source"] == root

    def test_a_short_key_finds_its_node(self):
        root = source("a.csv")

        assert Chain.of(clean(root))[root.key[:8]] == root

    def test_a_whole_key_finds_its_node(self):
        root = source("a.csv")

        assert Chain.of(clean(root))[root.key] == root

    def test_a_pasted_line_finds_its_node(self):
        root = source("a.csv")

        assert Chain.of(clean(root))[f"source {root.key[:8]}"] == root

    def test_a_mark_is_ignored(self):
        """What you copy out of `render(chain, "flat")` on a chain that has a runner."""
        root = source("a.csv")

        assert Chain.of(clean(root))[f"✓ source {root.key[:8]}"] == root

    def test_a_branch_is_ignored(self):
        """What you copy out of `render(chain, "tree")`, indentation and all."""
        root = source("a.csv")

        assert Chain.of(clean(root))[f"    └── ! source {root.key[:8]}"] == root

    def test_a_name_with_several_nodes_is_ambiguous(self):
        root = source("a.csv")
        built = Chain.of(merge(clean(root), clean(root, drop_empty=False)))

        with pytest.raises(AmbiguousTask, match="clean"):
            built["clean"]

    def test_the_ambiguity_lists_the_lines_that_resolve(self):
        """The error hands back the strings that disambiguate, ready to paste."""
        root = source("a.csv")
        first, second = clean(root), clean(root, drop_empty=False)
        built = Chain.of(merge(first, second))

        with pytest.raises(AmbiguousTask) as raised:
            built["clean"]

        assert f"clean {first.key[:8]}" in str(raised.value)
        assert f"clean {second.key[:8]}" in str(raised.value)

    def test_a_key_disambiguates_a_repeated_task(self):
        root = source("a.csv")
        second = clean(root, drop_empty=False)
        built = Chain.of(merge(clean(root), second))

        assert built[f"clean {second.key[:8]}"] == second

    def test_tokens_pointing_at_different_nodes_match_nothing(self):
        """Half of one line pasted onto half of another. No node answers both, so none is found."""
        root = source("a.csv")
        tail = clean(root)
        built = Chain.of(tail)

        with pytest.raises(KeyError):
            built[f"source {tail.key[:8]}"]

    def test_a_misspelled_name_beside_a_key_still_resolves(self):
        """The cost of dropping noise: a wrong name is noise, and the key carries the lookup.

        A wrong name on its own raises, so the typo is only ever silent when a key is there to
        answer in its place — and the key is the stronger half of the line.
        """
        root = source("a.csv")

        assert Chain.of(clean(root))[f"sauce {root.key[:8]}"] == root

    def test_an_unknown_string_raises(self):
        with pytest.raises(KeyError):
            Chain.of(source("a.csv"))["nonsense"]

    def test_an_empty_string_raises(self):
        with pytest.raises(KeyError):
            Chain.of(source("a.csv"))[""]

    def test_a_string_of_marks_alone_raises(self):
        with pytest.raises(KeyError):
            Chain.of(source("a.csv"))["└── ✓"]

    def test_the_task_path_keeps_its_return_type(self):
        """The typed lookup is why `__getitem__` takes a task. The string path must not cost it."""
        built = Chain.of(clean(source("a.csv")))

        assert_type(built[clean], Node[str])
        assert_type(built["clean"], Node[Any])


class TestInspection:
    def test_len_counts_the_nodes(self):
        assert len(Chain.of(clean(source("a.csv")))) == 2

    def test_membership_is_by_key(self):
        built = Chain.of(clean(source("a.csv")))

        assert source("a.csv") in built
        assert source("b.csv") not in built

    def test_printing_names_every_task(self):
        printed = str(Chain.of(merge(source("a.csv"), clean(source("b.csv")))))

        assert "source" in printed
        assert "clean" in printed
        assert "merge" in printed

    def test_printing_shows_the_key(self):
        node = source("a.csv")

        assert node.key[:8] in str(Chain.of(node))

    def test_printing_has_one_line_per_node(self):
        built = Chain.of(merge(source("a.csv"), source("b.csv")))

        assert len(str(built).strip().splitlines()) == len(built)


class TestDecorator:
    def test_a_bare_decorator_builds_a_chain(self):
        @chain
        def pipeline() -> Node[str]:
            return clean(source("a.csv"))

        assert isinstance(pipeline(), Chain)

    def test_the_decorated_function_keeps_its_parameters(self):
        @chain
        def pipeline(name: str) -> Node[str]:
            return clean(source(name))

        assert pipeline("a.csv").nodes != pipeline("b.csv").nodes

    def test_outputs_is_what_the_function_returned(self):
        @chain
        def pipeline() -> Node[str]:
            return clean(source("a.csv"))

        built = pipeline()

        assert built.outputs is built.terminals[0]

    def test_a_named_tuple_of_nodes_gives_several_terminals(self):
        class Both(NamedTuple):
            kept: Node[str]
            dropped: Node[str]

        @chain
        def pipeline() -> Both:
            root = source("a.csv")
            return Both(kept=clean(root), dropped=clean(root, drop_empty=False))

        built = pipeline()

        assert len(built.terminals) == 2
        assert built.outputs.kept in built
        assert len(built) == 3

    def test_building_a_chain_never_calls_a_task_body(self):
        calls: list[str] = []

        @task
        def recorded(path: str) -> str:
            calls.append(path)
            return path

        @chain
        def pipeline() -> Node[str]:
            return recorded("a.csv")

        pipeline()

        assert calls == []

    def test_printing_a_chain_never_calls_a_task_body(self):
        calls: list[str] = []

        @task
        def recorded(path: str) -> str:
            calls.append(path)
            return path

        @chain
        def pipeline() -> Node[str]:
            return recorded("a.csv")

        str(pipeline())

        assert calls == []


class TestRunnerBinding:
    def test_a_chain_without_a_runner_cannot_run(self):
        @chain
        def pipeline() -> Node[str]:
            return clean(source("a.csv"))

        with pytest.raises(NoRunner):
            pipeline().value(clean)

    def test_root_binds_a_runner(self, tmp_path: Path):
        @task
        def answer(seed: int) -> str:
            return f"answer-{seed}"

        @chain(root=tmp_path)
        def pipeline() -> Node[str]:
            return answer(seed=1)

        assert pipeline().value(answer) == "answer-1"

    def test_runner_binds_a_runner(self, tmp_path: Path):
        @task
        def answer(seed: int) -> str:
            return f"answer-{seed}"

        @chain(runner=Runner(root=tmp_path))
        def pipeline() -> Node[str]:
            return answer(seed=1)

        assert pipeline().value(answer) == "answer-1"

    def test_root_and_runner_together_are_refused(self, tmp_path: Path):
        with pytest.raises(TypeError):
            chain(root=tmp_path, runner=Runner(root=tmp_path))

    def test_on_rebinds_without_touching_the_chain(self, tmp_path: Path):
        @task
        def answer(seed: int) -> str:
            return f"answer-{seed}"

        @chain
        def pipeline() -> Node[str]:
            return answer(seed=1)

        built = pipeline()
        rebound = built.on(Runner(root=tmp_path))

        assert rebound.value(answer) == "answer-1"
        assert built.runner is None
        assert rebound.nodes == built.nodes

    def test_value_accepts_a_node_when_the_task_is_ambiguous(self, tmp_path: Path):
        @task
        def answer(seed: int) -> str:
            return f"answer-{seed}"

        @chain(root=tmp_path)
        def pipeline() -> tuple[Node[str], Node[str]]:
            return answer(seed=1), answer(seed=2)

        built = pipeline()

        with pytest.raises(AmbiguousTask):
            built.value(answer)
        assert built.value(built.outputs[1]) == "answer-2"

    def test_value_takes_a_pasted_line(self, tmp_path: Path):
        @task
        def answer(seed: int) -> str:
            return f"answer-{seed}"

        @chain(root=tmp_path)
        def pipeline() -> Node[str]:
            return answer(seed=1)

        built = pipeline()

        assert built.value(f"answer {built[answer].key[:4]}") == "answer-1"

    def test_value_takes_a_key_alone(self, tmp_path: Path):
        @task
        def answer(seed: int) -> str:
            return f"answer-{seed}"

        @chain(root=tmp_path)
        def pipeline() -> tuple[Node[str], Node[str]]:
            return answer(seed=1), answer(seed=2)

        built = pipeline()

        assert built.value(built.outputs[1].key[:8]) == "answer-2"

    def test_value_by_text_is_ambiguous_when_the_name_is(self, tmp_path: Path):
        @task
        def answer(seed: int) -> str:
            return f"answer-{seed}"

        @chain(root=tmp_path)
        def pipeline() -> tuple[Node[str], Node[str]]:
            return answer(seed=1), answer(seed=2)

        with pytest.raises(AmbiguousTask):
            pipeline().value("answer")

    def test_value_by_text_is_untyped(self, tmp_path: Path):
        """The task path keeps `T`; text cannot, which is the cost of addressing a node by text."""

        @task
        def answer(seed: int) -> str:
            return f"answer-{seed}"

        @chain(root=tmp_path)
        def pipeline() -> Node[str]:
            return answer(seed=1)

        built = pipeline()

        assert_type(built.value(answer), str)
        assert_type(built.value("answer"), Any)

    def test_run_materializes_every_terminal(self, tmp_path: Path):
        @task
        def answer(seed: int) -> str:
            return f"answer-{seed}"

        @chain(root=tmp_path)
        def pipeline() -> tuple[Node[str], Node[str]]:
            return answer(seed=1), answer(seed=2)

        built = pipeline()
        built.run()

        assert all(built.runner is not None and built.runner.store.exists(node) for node in built.terminals)


def test_resolve_is_reachable_from_the_package():
    assert resolve("plain") == "plain"
