"""Spec for the Node / TaskDef primitives.

These tests define the intended behaviour; the implementation follows them.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import pytest
from conftest import ensemble, raw_movies, taskdef

from ediechain.core.encode import EncodeError
from ediechain.core.node import Node, iter_nodes, map_nodes
from ediechain.core.task import TaskDef


def test_taskdef_accepts_a_normal_signature(raw: TaskDef) -> None:
    assert raw.name == "raw_movies"


def test_taskdef_rejects_var_positional() -> None:
    def bad(a: int, *rest: int) -> int: raise NotImplementedError

    with pytest.raises(TypeError):
        taskdef(bad)


def test_taskdef_rejects_var_keyword() -> None:
    def bad(a: int, **rest: int) -> int: raise NotImplementedError

    with pytest.raises(TypeError):
        taskdef(bad)


def test_bind_returns_a_node(raw: TaskDef) -> None:
    assert isinstance(raw.bind("imdb.csv"), Node)


def test_bind_raises_on_missing_argument(raw: TaskDef) -> None:
    with pytest.raises(TypeError):
        raw.bind()


def test_bind_raises_on_unknown_keyword(raw: TaskDef) -> None:
    with pytest.raises(TypeError):
        raw.bind("imdb.csv", nonsense=1)


def test_bind_normalizes_positional_to_keyword(filtered: TaskDef, raw: TaskDef) -> None:
    movies = raw.bind("imdb.csv")

    positional = filtered.bind(movies, 1960)
    keyword = filtered.bind(movies, from_year=1960)

    assert dict(positional.arguments) == dict(keyword.arguments)
    assert positional.key == keyword.key


def test_defaults_are_materialized_into_arguments(filtered: TaskDef, raw: TaskDef) -> None:
    node = filtered.bind(raw.bind("imdb.csv"))

    assert set(node.arguments) == {"raw", "from_year", "min_votes"}
    assert node.arguments["from_year"] == 1945


def test_arguments_are_read_only(raw: TaskDef) -> None:
    node = raw.bind("imdb.csv")

    with pytest.raises(TypeError):
        node.arguments["source"] = "other.csv"  # ty: ignore[invalid-assignment]


def test_identical_nodes_are_equal_and_share_a_key(raw: TaskDef) -> None:
    a = raw.bind("imdb.csv")
    b = raw.bind("imdb.csv")

    assert a.key == b.key
    assert a == b


def test_different_arguments_give_different_keys(raw: TaskDef) -> None:
    assert raw.bind("a.csv").key != raw.bind("b.csv").key


def test_nodes_are_frozen(raw: TaskDef) -> None:
    node = raw.bind("imdb.csv")

    with pytest.raises((AttributeError, TypeError)):
        node.key = "tampered"  # ty: ignore[invalid-assignment]


def test_identical_nodes_collapse_in_a_set(raw: TaskDef) -> None:
    assert len({raw.bind("imdb.csv"), raw.bind("imdb.csv"), raw.bind("other.csv")}) == 2


def test_nodes_work_as_dict_keys(raw: TaskDef) -> None:
    node = raw.bind("imdb.csv")

    assert {node: "value"}[raw.bind("imdb.csv")] == "value"


def test_version_participates_in_the_key() -> None:
    v1 = taskdef(raw_movies, version=1).bind("imdb.csv")
    v2 = taskdef(raw_movies, version=2).bind("imdb.csv")

    assert v1.key != v2.key


def test_task_name_participates_in_the_key() -> None:
    a = taskdef(raw_movies, name="a").bind("imdb.csv")
    b = taskdef(raw_movies, name="b").bind("imdb.csv")

    assert a.key != b.key


def test_passing_a_value_equal_to_the_default_keys_the_same(filtered: TaskDef, raw: TaskDef) -> None:
    movies = raw.bind("imdb.csv")

    omitted = filtered.bind(movies)
    explicit = filtered.bind(movies, from_year=1945)

    assert omitted.key == explicit.key


def test_changing_a_default_changes_the_key(raw: TaskDef) -> None:
    def before(raw: str, lr: float = 1e-6) -> str: raise NotImplementedError

    def after(raw: str, lr: float = 1e-5) -> str: raise NotImplementedError

    movies = raw.bind("imdb.csv")

    old = taskdef(before, name="model").bind(movies)
    new = taskdef(after, name="model").bind(movies)

    assert old.key != new.key


def test_adding_an_unmarked_parameter_changes_existing_keys(raw: TaskDef) -> None:
    def before(raw: str) -> str: raise NotImplementedError

    def after(raw: str, seed: int = 0) -> str: raise NotImplementedError

    movies = raw.bind("imdb.csv")

    old = taskdef(before, name="features").bind(movies)
    new = taskdef(after, name="features").bind(movies)

    assert old.key != new.key


def test_adding_a_transparent_parameter_does_not_change_existing_keys(raw: TaskDef) -> None:
    def before(raw: str) -> str: raise NotImplementedError

    def after(raw: str, seed: int = 0) -> str: raise NotImplementedError

    movies = raw.bind("imdb.csv")

    old = taskdef(before, name="features").bind(movies)
    new = taskdef(after, name="features", transparent={"seed"}).bind(movies)

    assert old.key == new.key


def test_transparent_parameter_counts_once_it_differs_from_its_default(raw: TaskDef) -> None:
    def features(raw: str, seed: int = 0) -> str: raise NotImplementedError

    task = taskdef(features, transparent={"seed"})
    movies = raw.bind("imdb.csv")

    assert task.bind(movies).key != task.bind(movies, seed=7).key


def test_ignored_parameter_never_reaches_the_key(raw: TaskDef) -> None:
    def features(raw: str, verbose: bool = False) -> str: raise NotImplementedError

    task = taskdef(features, ignore={"verbose"})
    movies = raw.bind("imdb.csv")

    assert task.bind(movies).key == task.bind(movies, verbose=True).key


def test_unknown_marked_parameter_is_rejected() -> None:
    def features(raw: str) -> str: raise NotImplementedError

    with pytest.raises(ValueError):
        taskdef(features, ignore={"typo"})

    with pytest.raises(ValueError):
        taskdef(features, transparent={"typo"})


def test_iter_nodes_finds_a_bare_node(raw: TaskDef) -> None:
    node = raw.bind("imdb.csv")

    assert list(iter_nodes(node)) == [node]


def test_iter_nodes_finds_nodes_in_a_list(raw: TaskDef) -> None:
    a, b = raw.bind("a.csv"), raw.bind("b.csv")

    assert set(iter_nodes([a, b])) == {a, b}


def test_iter_nodes_finds_nodes_nested_in_a_dict_of_lists(raw: TaskDef) -> None:
    a, b = raw.bind("a.csv"), raw.bind("b.csv")

    assert set(iter_nodes({"models": [a], "extra": {"deep": b}})) == {a, b}


def test_iter_nodes_ignores_plain_values() -> None:
    assert list(iter_nodes({"a": 1, "b": ["x", "y"], "c": None})) == []


def test_iter_nodes_does_not_recurse_into_strings() -> None:
    assert list(iter_nodes("abc")) == []


def test_map_nodes_replaces_nodes_in_place(raw: TaskDef) -> None:
    a, b = raw.bind("a.csv"), raw.bind("b.csv")

    result = map_nodes({"models": [a, b]}, lambda n: n.key)

    assert result == {"models": [a.key, b.key]}


def test_map_nodes_preserves_container_types(raw: TaskDef) -> None:
    node = raw.bind("a.csv")

    result = map_nodes({"t": (node,), "l": [node]}, lambda n: "X")

    assert result == {"t": ("X",), "l": ["X"]}
    assert isinstance(result["t"], tuple)
    assert isinstance(result["l"], list)


def test_map_nodes_leaves_plain_values_untouched() -> None:
    value = {"a": 1, "b": ["x"], "c": None}

    assert map_nodes(value, lambda n: "X") == value


def test_node_inputs_are_flat_and_deduped(raw: TaskDef) -> None:
    a = raw.bind("a.csv")
    node = taskdef(ensemble).bind([a, a], weights={"a": 1.0})

    assert tuple(node.inputs) == (a,)


def test_nested_nodes_participate_in_the_key(raw: TaskDef) -> None:
    ens = taskdef(ensemble)

    one = ens.bind([raw.bind("a.csv")])
    other = ens.bind([raw.bind("b.csv")])

    assert one.key != other.key


def test_upstream_change_propagates_downstream(raw: TaskDef, filtered: TaskDef) -> None:
    downstream_a = filtered.bind(raw.bind("a.csv"))
    downstream_b = filtered.bind(raw.bind("b.csv"))

    assert downstream_a.key != downstream_b.key


def test_sibling_nodes_are_unaffected_by_an_unrelated_change(raw: TaskDef, filtered: TaskDef) -> None:
    movies = raw.bind("imdb.csv")

    sibling = filtered.bind(movies, min_votes=10)
    before = filtered.bind(movies, from_year=1960)
    after = filtered.bind(movies, from_year=1970)

    assert before.key != after.key
    assert sibling.key == filtered.bind(movies, min_votes=10).key


def test_identical_subgraphs_built_separately_share_keys(raw: TaskDef, filtered: TaskDef) -> None:
    one = filtered.bind(raw.bind("imdb.csv"), from_year=1960)
    other = filtered.bind(raw.bind("imdb.csv"), from_year=1960)

    assert one.key == other.key


def test_tuple_and_list_arguments_do_not_collide() -> None:
    ens = taskdef(ensemble)

    assert ens.bind((1, 2)).key != ens.bind([1, 2]).key


def test_int_and_str_dict_keys_do_not_collide() -> None:
    ens = taskdef(ensemble)

    assert ens.bind([], weights={1: 1.0}).key != ens.bind([], weights={"1": 1.0}).key


def test_unencodable_argument_raises_a_clear_error() -> None:
    ens = taskdef(ensemble)

    with pytest.raises(EncodeError, match="cannot encode"):
        ens.bind([object()])


def test_an_ignored_parameter_may_hold_an_unencodable_value() -> None:
    """`ignore` keeps the value out of the key, so it never reaches the encoder.

    This is how a logger, a client or a connection is passed to a task.
    """

    def features(raw: str, client: Any = None) -> str: raise NotImplementedError

    task = taskdef(features, ignore={"client"})

    assert task.bind("a.csv", client=object()).key == task.bind("a.csv").key


def test_a_transparent_default_is_compared_on_encoded_form_not_equality() -> None:
    """`==` on an array returns an array, and `if` on it raises. Compare the encodings."""
    numpy = pytest.importorskip("numpy")

    def features(raw: str, weights: Any = (1.0, 2.0)) -> str: raise NotImplementedError

    task = taskdef(features, transparent={"weights"})

    assert task.bind("a.csv", weights=numpy.array([1.0, 2.0])).key != task.bind("a.csv").key


def test_a_node_hidden_from_the_traversal_is_rejected(raw: TaskDef) -> None:
    """A node inside a container `map_nodes` does not walk would be hashed as a value and never
    scheduled as an input. It must fail loudly instead.
    """
    movies = raw.bind("imdb.csv")

    with pytest.raises(EncodeError):
        taskdef(ensemble).bind([], weights={"models": {movies}})


def test_a_transparent_parameter_may_hold_nodes_inside_a_container(raw: TaskDef) -> None:
    def features(raw: str, extra: tuple[str, ...] = ()) -> str: raise NotImplementedError

    task = taskdef(features, transparent={"extra"})
    movies = raw.bind("imdb.csv")

    assert task.bind(movies, extra=[raw.bind("extra.csv")]).key != task.bind(movies).key


class Point(NamedTuple):
    x: int
    y: int


class Size(NamedTuple):
    width: int
    height: int


def test_named_tuples_of_different_classes_give_different_keys() -> None:
    def place(at: Any) -> str: raise NotImplementedError

    task = taskdef(place)

    assert task.bind(Point(1, 2)).key != task.bind(Size(1, 2)).key
