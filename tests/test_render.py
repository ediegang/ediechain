"""Spec for `render(chain, view=...)`.

Rendering is not core: a chain computes and stores without it, and `Chain.__str__` stays the plain
topological list it has always been. Everything that reads state goes through `render`.

Three views over one chain, because a tree cannot draw a DAG and only one of them has to. `flat`
and `mermaid` run source -> terminal; `tree` runs terminal -> sources, so a node with two consumers
costs one reference line instead of a dead end.

Every view is a `str` and every view reads the store without running a body. State comes from the
chain's own runner; with no runner there is no state to show, so the mark is absent and the view
still renders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import pytest

from ediechain import IN_MEMORY, Chain, In, Node, Runner, chain, render, resolve, task

STORED, PENDING, MEMORY, FORCED, REFERENCE = "✓", "·", "○", "!", "↑"

CALLS: list[str] = []


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
def pair(first: In[int], second: In[int]) -> int:
    CALLS.append("pair")
    return resolve(first) + resolve(second)


class Outputs(NamedTuple):
    first: Node[int]
    second: Node[int]


@pytest.fixture(autouse=True)
def _reset():
    CALLS.clear()


@pytest.fixture
def runner(tmp_path: Path) -> Runner:
    return Runner(root=tmp_path)


@pytest.fixture
def line(runner: Runner) -> Chain[Any]:
    """Three nodes, no convergence: source -> middle -> leaf."""
    return Chain.of(leaf(middle(source(seed=1)))).on(runner)


@pytest.fixture
def converging(runner: Runner) -> Chain[Any]:
    """`middle` has two consumers, so a tree must reference it once. Five nodes, one reference."""
    base = middle(source(seed=1))
    return Chain.of(pair(leaf(base), memo(base))).on(runner)


def body(rendered: str) -> list[str]:
    """One token per line: the task name. Survives indentation, branches and the mark column."""
    return [text.split()[-2] for text in rendered.splitlines()]


def marks(rendered: str) -> list[str]:
    return [text.split()[-3] for text in rendered.splitlines()]


class TestSurface:
    def test_the_default_view_is_the_tree(self, line: Chain[Any]):
        """`flat` is already what `print(chain)` gives, so the default is the one you cannot get."""
        assert render(line) == render(line, "tree")

    def test_an_unknown_view_is_refused(self, line: Chain[Any]):
        with pytest.raises(ValueError, match="unknown view"):
            render(line, "graph")  # ty: ignore[invalid-argument-type]

    def test_the_refusal_names_the_views(self, line: Chain[Any]):
        with pytest.raises(ValueError, match="flat.*tree.*mermaid"):
            render(line, "graph")  # ty: ignore[invalid-argument-type]

    def test_printing_a_chain_shows_no_state(self, converging: Chain[Any]):
        """The core does not render. `__str__` is the plain list, with no mark and no store read."""
        assert all(len(text.split()) == 2 for text in str(converging).splitlines())

    def test_printing_a_chain_is_not_a_view(self, converging: Chain[Any]):
        assert str(converging) != render(converging, "flat")

    def test_a_chain_has_no_markdown_repr(self, converging: Chain[Any]):
        """Mermaid reaches a notebook as `mo.mermaid(render(chain, "mermaid"))`, not through `Chain`."""
        assert not hasattr(converging, "_repr_markdown_")


class TestMarks:
    def test_a_node_in_the_store_is_ticked(self, line: Chain[Any], runner: Runner):
        """Computing `middle` stores `source` with it, and leaves `leaf` to compute."""
        runner.value(line[middle])

        assert marks(render(line, "flat")) == [STORED, STORED, PENDING]

    def test_a_node_not_yet_computed_is_dotted(self, line: Chain[Any]):
        assert set(marks(render(line, "flat"))) == {PENDING}

    def test_a_never_persisted_node_is_hollow(self, converging: Chain[Any]):
        rendered = render(converging, "flat")

        assert marks(rendered)[body(rendered).index("memo")] == MEMORY

    def test_forced_beats_stored(self, line: Chain[Any], runner: Runner):
        runner.value(line[leaf])
        rendered = render(line.force(middle), "flat")

        assert marks(rendered)[body(rendered).index("middle")] == FORCED

    def test_forced_beats_never_persisted(self, converging: Chain[Any]):
        """A forced `IN_MEMORY` node marks `!`, not `○`.

        The mark exists to show what reruns. `○` in the middle of a forced path breaks the one
        thing the view answers, and `never on disk` is still readable from the absence of a tick
        anywhere else in the chain.
        """
        rendered = render(converging.force(memo), "flat")

        assert marks(rendered)[body(rendered).index("memo")] == FORCED

    def test_the_whole_cascade_is_marked(self, line: Chain[Any], runner: Runner):
        runner.value(line[leaf])

        assert set(marks(render(line.force(source), "flat"))) == {FORCED}

    def test_forcing_does_not_mark_upstream(self, line: Chain[Any], runner: Runner):
        runner.value(line[leaf])
        rendered = render(line.force(middle), "flat")

        assert marks(rendered)[body(rendered).index("source")] == STORED

    def test_a_chain_with_no_runner_shows_no_mark(self):
        rendered = render(Chain.of(leaf(middle(source(seed=1)))), "flat")

        assert all(len(text.split()) == 2 for text in rendered.splitlines())

    def test_a_chain_with_no_runner_renders_the_same_shape(self):
        built = Chain.of(leaf(middle(source(seed=1))))

        assert body(render(built, "flat")) == [node.task.name for node in built]

    def test_rendering_runs_no_body(self, converging: Chain[Any]):
        render(converging, "flat")
        render(converging, "tree")
        render(converging, "mermaid")

        assert CALLS == []

    def test_rendering_a_warm_chain_runs_no_body(self, line: Chain[Any], runner: Runner):
        runner.value(line[leaf])
        CALLS.clear()

        render(line, "flat")

        assert CALLS == []


class TestFlat:
    def test_one_line_per_node(self, converging: Chain[Any]):
        assert len(render(converging, "flat").splitlines()) == len(converging)

    def test_the_order_is_the_chain_s_order(self, converging: Chain[Any]):
        assert body(render(converging, "flat")) == [node.task.name for node in converging]

    def test_every_line_carries_the_short_key(self, line: Chain[Any]):
        rendered = render(line, "flat")

        for node in line:
            assert node.key[:8] in rendered

    def test_a_node_appears_once(self, converging: Chain[Any]):
        assert body(render(converging, "flat")).count("middle") == 1

    def test_there_is_no_reference_line(self, converging: Chain[Any]):
        """Flat lists each node once, so `↑` belongs to the tree alone."""
        assert REFERENCE not in render(converging, "flat")

    def test_the_line_format_is_mark_name_key(self, line: Chain[Any]):
        first = render(line, "flat").splitlines()[0]
        node = line.nodes[0]

        assert first == f"{PENDING} {node.task.name} {node.key[:8]}"

    def test_the_line_format_without_a_runner_is_name_key(self):
        built = Chain.of(source(seed=1))
        node = built.nodes[0]

        assert render(built, "flat") == f"{node.task.name} {node.key[:8]}"


class TestTree:
    def test_the_root_is_the_terminal(self, line: Chain[Any]):
        assert body(render(line, "tree"))[0] == "leaf"

    def test_a_dependency_sits_under_its_consumer(self, line: Chain[Any]):
        assert body(render(line, "tree")) == ["leaf", "middle", "source"]

    def test_a_dependency_is_indented_further(self, line: Chain[Any]):
        indents = [len(text) - len(text.lstrip()) for text in render(line, "tree").splitlines()]

        assert indents == sorted(indents) and indents[0] < indents[-1]

    def test_a_chain_without_convergence_has_one_line_per_node(self, line: Chain[Any]):
        assert len(render(line, "tree").splitlines()) == len(line)

    def test_a_shared_node_is_drawn_once(self, converging: Chain[Any]):
        drawn = [text for text in render(converging, "tree").splitlines() if REFERENCE not in text]

        assert [text.split()[-2] for text in drawn].count("middle") == 1

    def test_a_shared_node_becomes_a_reference(self, converging: Chain[Any]):
        references = [text for text in render(converging, "tree").splitlines() if REFERENCE in text]

        assert len(references) == 1
        assert references[0].split()[-2] == "middle"

    def test_a_reference_keeps_its_state_mark(self, converging: Chain[Any], runner: Runner):
        runner.value(converging[middle])
        reference = next(text for text in render(converging, "tree").splitlines() if REFERENCE in text)

        assert reference.split()[-3] == STORED

    def test_a_reference_repeats_no_subtree(self, converging: Chain[Any]):
        """`source` is `middle`'s input. It is drawn once, under the `middle` that was drawn."""
        assert body(render(converging, "tree")).count("source") == 1

    def test_the_line_count_is_nodes_plus_references(self, converging: Chain[Any]):
        rendered = render(converging, "tree")
        references = sum(REFERENCE in text for text in rendered.splitlines())

        assert len(rendered.splitlines()) == len(converging) + references

    def test_named_outputs_head_their_roots(self):
        @chain
        def experiment() -> Outputs:
            base = middle(source(seed=1))
            return Outputs(first=leaf(base), second=memo(base))

        rendered = render(experiment(), "tree")

        assert rendered.startswith("first: ")
        assert "\nsecond: " in rendered

    def test_unnamed_outputs_have_no_heading(self, converging: Chain[Any]):
        assert ":" not in render(converging, "tree")

    def test_a_chain_with_no_runner_still_references(self):
        base = middle(source(seed=1))
        rendered = render(Chain.of(pair(leaf(base), memo(base))), "tree")

        assert sum(REFERENCE in text for text in rendered.splitlines()) == 1


class TestMermaid:
    def test_it_is_a_top_down_graph(self, line: Chain[Any]):
        assert render(line, "mermaid").splitlines()[0] == "graph TD"

    def test_one_declaration_per_node(self, converging: Chain[Any]):
        declarations = [text for text in render(converging, "mermaid").splitlines() if "-->" not in text]

        assert len(declarations) == len(converging) + 1  # + the `graph TD` header

    def test_an_edge_runs_source_to_terminal(self, line: Chain[Any]):
        rendered = render(line, "mermaid")

        assert f"n{line[source].key[:8]} --> n{line[middle].key[:8]}" in rendered

    def test_every_input_is_an_edge(self, converging: Chain[Any]):
        edges = [text for text in render(converging, "mermaid").splitlines() if "-->" in text]

        assert len(edges) == sum(len(node.inputs) for node in converging)

    def test_convergence_is_drawn(self, converging: Chain[Any]):
        """What the tree cannot do: one node with two consumers, as two edges out of one id."""
        shared = f"n{converging[middle].key[:8]} -->"

        assert render(converging, "mermaid").count(shared) == 2

    def test_an_id_is_safe(self, converging: Chain[Any]):
        """A key is hex, so a bare key can start with a digit. Mermaid ids may not."""
        declarations = [text for text in render(converging, "mermaid").splitlines() if "-->" not in text][1:]

        for text in declarations:
            assert text.strip()[0] == "n"

    def test_a_label_carries_name_key_and_mark(self, line: Chain[Any]):
        rendered = render(line, "mermaid")

        assert f'"{PENDING} {line[source].task.name}<br/>{line[source].key[:8]}"' in rendered

    def test_a_label_without_a_runner_carries_no_mark(self):
        built = Chain.of(source(seed=1))
        node = built.nodes[0]

        assert f'"{node.task.name}<br/>{node.key[:8]}"' in render(built, "mermaid")

    def test_a_quote_in_a_task_name_is_escaped(self):
        @task(name='weird "name"')
        def oddly_named(seed: int) -> int:
            raise NotImplementedError

        rendered = render(Chain.of(oddly_named(seed=1)), "mermaid")
        label = rendered.splitlines()[1].split("[", 1)[1]

        assert label.count('"') == 2
        assert "#quot;" in label

    def test_a_bracket_in_a_task_name_survives(self):
        @task(name="odd[1]")
        def bracketed(seed: int) -> int:
            raise NotImplementedError

        rendered = render(Chain.of(bracketed(seed=1)), "mermaid")

        assert rendered.splitlines()[1].endswith("]")
