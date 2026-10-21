"""Spec for the `@task` decorator, `In` and `resolve`.

A task parameter has two types. At the call site it is a `Node[T]`, because the artifact does not
exist yet. Inside the body it is a `T`, because the executor substituted it. `In[T]` is the
annotation that admits both, and `resolve` narrows it back to `T` for the body.
"""

from __future__ import annotations

import pytest

from ediechain.core.node import Node
from ediechain.core.task import In, resolve, task


@task
def source(path: str) -> str:
    raise NotImplementedError


class TestDecoratorForms:
    def test_a_bare_decorator_builds_a_task(self):
        @task
        def clean(raw: In[str]) -> str:
            raise NotImplementedError

        assert isinstance(clean("a.csv"), Node)

    def test_the_decorator_accepts_arguments(self):
        @task(version=2)
        def clean(raw: In[str]) -> str:
            raise NotImplementedError

        assert isinstance(clean("a.csv"), Node)

    def test_wiring_does_not_call_the_body(self):
        calls: list[str] = []

        @task
        def clean(raw: str) -> str:
            calls.append(raw)
            return raw

        clean("a.csv")

        assert calls == []


class TestIdentity:
    def test_the_task_name_defaults_to_the_function_name(self):
        @task
        def clean(raw: In[str]) -> str:
            raise NotImplementedError

        assert clean("a.csv").task.name == "clean"

    def test_name_overrides_the_function_name(self):
        @task(name="movies:clean")
        def clean(raw: In[str]) -> str:
            raise NotImplementedError

        assert clean("a.csv").task.name == "movies:clean"

    def test_the_name_participates_in_the_key(self):
        @task(name="one")
        def first(raw: In[str]) -> str:
            raise NotImplementedError

        @task(name="other")
        def second(raw: In[str]) -> str:
            raise NotImplementedError

        assert first("a.csv").key != second("a.csv").key

    def test_where_the_function_is_defined_does_not_reach_the_key(self):
        """Decision 15, one level up. A file move must not orphan the data."""

        def build_one():
            @task
            def clean(raw: In[str]) -> str:
                raise NotImplementedError

            return clean("a.csv")

        def build_other():
            @task
            def clean(raw: In[str]) -> str:
                raise NotImplementedError

            return clean("a.csv")

        assert build_one().key == build_other().key


class TestForwarding:
    def test_version_reaches_the_key(self):
        @task(version=1)
        def one(raw: In[str]) -> str:
            raise NotImplementedError

        @task(version=2, name="one")
        def two(raw: In[str]) -> str:
            raise NotImplementedError

        assert one("a.csv").key != two("a.csv").key

    def test_an_ignored_parameter_never_reaches_the_key(self):
        @task(ignore={"verbose"})
        def clean(raw: In[str], verbose: bool = False) -> str:
            raise NotImplementedError

        assert clean("a.csv").key == clean("a.csv", verbose=True).key

    def test_a_transparent_parameter_drops_out_at_its_default(self):
        @task(transparent={"seed"})
        def with_seed(raw: In[str], seed: int = 0) -> str:
            raise NotImplementedError

        @task(name="with_seed")
        def without_seed(raw: In[str]) -> str:
            raise NotImplementedError

        assert with_seed("a.csv").key == without_seed("a.csv").key
        assert with_seed("a.csv").key != with_seed("a.csv", seed=7).key

    def test_an_unknown_marked_parameter_is_rejected_at_decoration_time(self):
        with pytest.raises(ValueError):

            @task(ignore={"typo"})
            def clean(raw: In[str]) -> str:
                raise NotImplementedError

    def test_var_args_are_rejected_at_decoration_time(self):
        with pytest.raises(TypeError):

            @task
            def clean(raw: In[str], *rest: str) -> str:
                raise NotImplementedError

    def test_var_keywords_are_rejected_at_decoration_time(self):
        with pytest.raises(TypeError):

            @task
            def clean(raw: In[str], **rest: str) -> str:
                raise NotImplementedError


class TestBinding:
    def test_positional_becomes_keyword(self):
        @task
        def clean(raw: In[str], from_year: int = 1945) -> str:
            raise NotImplementedError

        assert clean("a.csv", 1960).key == clean("a.csv", from_year=1960).key

    def test_defaults_are_materialized(self):
        @task
        def clean(raw: In[str], from_year: int = 1945) -> str:
            raise NotImplementedError

        assert set(clean("a.csv").arguments) == {"raw", "from_year"}

    def test_a_missing_argument_raises(self):
        @task
        def clean(raw: In[str]) -> str:
            raise NotImplementedError

        with pytest.raises(TypeError):
            clean()  # ty: ignore[missing-argument]

    def test_an_unknown_keyword_raises(self):
        @task
        def clean(raw: In[str]) -> str:
            raise NotImplementedError

        with pytest.raises(TypeError):
            clean("a.csv", nonsense=1)  # ty: ignore[unknown-argument]

    def test_a_node_is_accepted_as_an_argument(self):
        @task
        def clean(raw: In[str]) -> str:
            raise NotImplementedError

        movies = source("a.csv")

        assert clean(movies).inputs == (movies,)


class TestReturnAnnotation:
    def test_the_return_annotation_is_resolved_to_a_type(self):
        """Storage dispatches on `returns`. Under `from __future__ import annotations` the raw
        signature holds the string `"str"`, which no handler table can be keyed on.
        """

        @task
        def clean(raw: In[str]) -> str:
            raise NotImplementedError

        assert clean("a.csv").task.returns is str

    def test_a_task_without_a_return_annotation_is_rejected(self):
        """Storage has nothing to dispatch on. Say so at import, not at write time."""
        with pytest.raises(TypeError, match="return annotation"):

            @task
            def clean(raw):
                raise NotImplementedError

    def test_a_task_returning_none_is_allowed(self):
        """`-> None` is a real answer; a missing annotation is not. They are distinguishable."""

        @task
        def report(raw: In[str]) -> None:
            raise NotImplementedError

        assert report("a.csv").task.returns is type(None)


class TestDirectCall:
    def test_fn_is_the_original_function(self):
        def body(raw: str) -> str:
            return raw.upper()

        wired = task(body)

        assert wired.fn is body

    def test_fn_runs_the_body(self):
        @task
        def clean(raw: str) -> str:
            return raw.upper()

        assert clean.fn("a.csv") == "A.CSV"


class TestResolve:
    def test_resolve_returns_a_plain_value_unchanged(self):
        value = ["a", "b"]

        assert resolve(value) is value

    def test_resolve_refuses_a_node(self):
        """The executor substitutes artifacts before the body runs. A node here means it did not."""
        with pytest.raises(TypeError):
            resolve(source("a.csv"))
