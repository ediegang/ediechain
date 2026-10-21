"""Spec for `by_init_params`, the opt-in replacement for `AutoSlugname`.

Registered once on a base class, it reads the `__init__` parameters of the **concrete** class off
the instance. A subclass that adds a parameter is covered without its own registration.

The old base class did this by inheritance, for every subclass, whether or not the author wanted
it. Here it is one `register` call, and the failure lands where someone chose to put it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from ediechain.core.converters import by_init_params
from ediechain.core.encode import EncodeError, encode
from ediechain.core.registry import register


class TestParameters:
    def test_init_parameters_become_key_material(self):
        class Model:
            def __init__(self, lr: float) -> None:
                self.lr = lr

        register(Model)(by_init_params())

        assert encode(Model(1e-3)) != encode(Model(1e-4))

    def test_equal_instances_encode_the_same(self):
        class Model:
            def __init__(self, lr: float, layers: int) -> None:
                self.lr, self.layers = lr, layers

        register(Model)(by_init_params())

        assert encode(Model(1e-3, 4)) == encode(Model(1e-3, 4))

    def test_self_is_never_read(self):
        """There is no attribute named `self`, so reading it would raise."""

        class Model:
            def __init__(self) -> None: ...

        register(Model)(by_init_params())

        assert encode(Model()) == encode(Model())

    def test_keyword_only_parameters_participate(self):
        class Model:
            def __init__(self, lr: float, *, seed: int = 0) -> None:
                self.lr, self.seed = lr, seed

        register(Model)(by_init_params())

        assert encode(Model(1e-3, seed=1)) != encode(Model(1e-3, seed=2))

    def test_var_args_are_skipped(self):
        """`*args` and `**kwargs` are signature names, not attributes. Reading them would raise."""

        class Model:
            def __init__(self, lr: float, *rest: int, **options: int) -> None:
                self.lr = lr

        register(Model)(by_init_params())

        assert encode(Model(1e-3)) == encode(Model(1e-3))

    def test_a_class_without_an_init_has_no_key_material(self):
        class Marker:
            pass

        register(Marker)(by_init_params())

        assert encode(Marker()) == encode(Marker())


class TestExclusions:
    def test_an_excluded_parameter_never_reaches_the_key(self):
        class Model:
            def __init__(self, lr: float, verbose: bool = False) -> None:
                self.lr, self.verbose = lr, verbose

        register(Model)(by_init_params(exclude={"verbose"}))

        assert encode(Model(1e-3, verbose=True)) == encode(Model(1e-3, verbose=False))

    def test_exclude_takes_any_iterable(self):
        class Model:
            def __init__(self, lr: float, verbose: bool = False, cache_dir: str = "cache") -> None:
                self.lr, self.verbose, self.cache_dir = lr, verbose, cache_dir

        register(Model)(by_init_params(exclude=["verbose", "cache_dir"]))

        assert encode(Model(1e-3, verbose=True, cache_dir="other")) == encode(Model(1e-3))

    def test_private_parameters_are_skipped_by_default(self):
        class Model:
            def __init__(self, lr: float, _internal: int = 0) -> None:
                self.lr, self._internal = lr, _internal

        register(Model)(by_init_params())

        assert encode(Model(1e-3, _internal=1)) == encode(Model(1e-3, _internal=2))

    def test_private_parameters_can_be_kept(self):
        class Model:
            def __init__(self, lr: float, _internal: int = 0) -> None:
                self.lr, self._internal = lr, _internal

        register(Model)(by_init_params(skip_private=False))

        assert encode(Model(1e-3, _internal=1)) != encode(Model(1e-3, _internal=2))


class TestTransparent:
    """`transparent` keeps a parameter out of the key while it holds its default.

    This replaces adding a new option through `**kwargs` under `AutoSlugname`: old keys survive,
    an explicit default keys the same as an omitted one, and a misspelled name is a `TypeError`.
    """

    def test_a_transparent_parameter_at_its_default_stays_out_of_the_key(self):
        class Model:
            def __init__(self, lr: float, temperature: float = 1.0) -> None:
                self.lr, self.temperature = lr, temperature

        convert = by_init_params(transparent=["temperature"])

        assert convert(Model(1e-3)) == {"lr": 1e-3}

    def test_adding_a_transparent_parameter_keeps_the_old_key(self):
        """Compared on the converter output: one test cannot register two classes under one tag."""

        class Before:
            def __init__(self, lr: float) -> None:
                self.lr = lr

        class After:
            def __init__(self, lr: float, temperature: float = 1.0) -> None:
                self.lr, self.temperature = lr, temperature

        convert = by_init_params(transparent=["temperature"])

        assert convert(After(1e-3)) == convert(Before(1e-3))

    def test_a_transparent_parameter_off_its_default_enters_the_key(self):
        class Model:
            def __init__(self, lr: float, temperature: float = 1.0) -> None:
                self.lr, self.temperature = lr, temperature

        register(Model)(by_init_params(transparent=["temperature"]))

        assert encode(Model(1e-3, temperature=0.5)) != encode(Model(1e-3))

    def test_an_explicit_default_keys_the_same_as_an_omitted_one(self):
        """`**kwargs` under `AutoSlugname` failed this: `{}` and `{"temperature": 1.0}` differ."""

        class Model:
            def __init__(self, lr: float, temperature: float = 1.0) -> None:
                self.lr, self.temperature = lr, temperature

        register(Model)(by_init_params(transparent=["temperature"]))

        assert encode(Model(1e-3, temperature=1.0)) == encode(Model(1e-3))

    def test_a_non_transparent_parameter_at_its_default_stays_in_the_key(self):
        class Model:
            def __init__(self, lr: float, seed: int = 0) -> None:
                self.lr, self.seed = lr, seed

        convert = by_init_params(transparent=["temperature"])

        assert convert(Model(1e-3)) == {"lr": 1e-3, "seed": 0}

    def test_a_parameter_without_a_default_always_enters_the_key(self):
        class Model:
            def __init__(self, lr: float) -> None:
                self.lr = lr

        convert = by_init_params(transparent=["lr"])

        assert convert(Model(1e-3)) == {"lr": 1e-3}

    def test_defaults_are_compared_by_encoding_not_by_equality(self):
        """`array == None` is elementwise, and its truth value raises."""

        class Model:
            def __init__(self, weights: np.ndarray | None = None) -> None:
                self.weights = weights

        convert = by_init_params(transparent=["weights"])

        assert convert(Model()) == {}
        assert "weights" in convert(Model(np.array([1, 2])))

    def test_transparent_holds_down_the_hierarchy(self):
        class Base:
            def __init__(self, lr: float, temperature: float = 1.0) -> None:
                self.lr, self.temperature = lr, temperature

        class Ridge(Base):
            def __init__(self, lr: float, alpha: float, temperature: float = 1.0) -> None:
                super().__init__(lr, temperature)
                self.alpha = alpha

        register(Base)(by_init_params(transparent=["temperature"]))

        assert encode(Ridge(1.0, 0.1, temperature=1.0)) == encode(Ridge(1.0, 0.1))
        assert encode(Ridge(1.0, 0.1, temperature=0.5)) != encode(Ridge(1.0, 0.1))

    def test_the_most_derived_default_wins(self):
        """The MRO walk visits `Base` after `Ridge`. Comparing against the last default seen would
        put `Ridge()` in the key and drop `Ridge(temperature=1.0)`, the opposite of the rule.
        """

        class Base:
            def __init__(self, temperature: float = 1.0) -> None:
                self.temperature = temperature

        class Ridge(Base):
            def __init__(self, temperature: float = 0.5) -> None:
                super().__init__(temperature)

        convert = by_init_params(transparent=["temperature"])

        assert convert(Ridge()) == {}
        assert convert(Ridge(temperature=1.0)) == {"temperature": 1.0}

    def test_a_dataclass_field_default_is_the_default(self):
        @dataclass(frozen=True, kw_only=True)
        class Model:
            lr: float
            temperature: float = 1.0

        convert = by_init_params(transparent=["temperature"])

        assert convert(Model(lr=1e-3)) == {"lr": 1e-3}

    def test_a_default_factory_field_stays_in_the_key(self):
        """The signature default is a `<factory>` sentinel, which does not encode. There is no
        default to compare with, so the field is treated like one without a default: the safe side.
        """

        @dataclass(frozen=True, kw_only=True)
        class Model:
            labels: list[str] = field(default_factory=list)

        convert = by_init_params(transparent=["labels"])

        assert convert(Model()) == {"labels": []}


class TestInheritance:
    def test_one_registration_covers_the_hierarchy(self):
        class Base:
            def __init__(self, lr: float) -> None:
                self.lr = lr

        class Ridge(Base):
            def __init__(self, lr: float, alpha: float) -> None:
                super().__init__(lr)
                self.alpha = alpha

        register(Base)(by_init_params())

        assert encode(Ridge(1.0, 0.1)) == encode(Ridge(1.0, 0.1))

    def test_a_parameter_added_by_a_subclass_participates(self):
        """The whole point. A hand-written converter on the base would collide here."""

        class Base:
            def __init__(self, lr: float) -> None:
                self.lr = lr

        class Ridge(Base):
            def __init__(self, lr: float, alpha: float) -> None:
                super().__init__(lr)
                self.alpha = alpha

        register(Base)(by_init_params())

        assert encode(Ridge(1.0, 0.1)) != encode(Ridge(1.0, 0.9))

    def test_a_base_and_a_subclass_do_not_collide(self):
        class Base:
            def __init__(self, lr: float) -> None:
                self.lr = lr

        class Ridge(Base):
            def __init__(self, lr: float) -> None:
                super().__init__(lr)

        register(Base)(by_init_params())

        assert encode(Base(1.0)) != encode(Ridge(1.0))

    def test_exclusions_hold_down_the_hierarchy(self):
        class Base:
            def __init__(self, lr: float, verbose: bool = False) -> None:
                self.lr, self.verbose = lr, verbose

        class Ridge(Base):
            def __init__(self, lr: float, alpha: float, verbose: bool = False) -> None:
                super().__init__(lr, verbose)
                self.alpha = alpha

        register(Base)(by_init_params(exclude={"verbose"}))

        assert encode(Ridge(1.0, 0.1, verbose=True)) == encode(Ridge(1.0, 0.1))


class TestFailsLoudly:
    def test_a_parameter_with_no_attribute_raises(self):
        """`AutoSlugname` raised here too. The message must name the missing attribute."""

        class Model:
            def __init__(self, lr: float, layers: int) -> None:
                self.lr = lr

        register(Model)(by_init_params())

        with pytest.raises(AttributeError, match="layers"):
            encode(Model(1e-3, 4))

    def test_an_unencodable_attribute_raises(self):
        class Model:
            def __init__(self, lr: float, client: object) -> None:
                self.lr, self.client = lr, client

        register(Model)(by_init_params())

        with pytest.raises(EncodeError):
            encode(Model(1e-3, object()))

    def test_a_property_not_named_in_init_is_never_read(self):
        class Model:
            def __init__(self, lr: float) -> None:
                self.lr = lr

            @property
            def slugname(self) -> str:
                raise AssertionError("the converter read an attribute it was not told about")

        register(Model)(by_init_params())

        assert encode(Model(1e-3)) == encode(Model(1e-3))


class TestStateNotInNamedAttributes:
    """`by_init_params` assumes a parameter name is also an attribute name.

    A base that swallows `*args, **kwargs` and stores them breaks that assumption: the subclass
    declares `name`, but the only attributes are `args` and `kwargs`. Register the base instead.
    """

    def test_by_init_params_raises_when_a_parameter_is_not_an_attribute(self):
        class Core:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args, self.kwargs = args, kwargs

        class Model(Core):
            def __init__(self, name: str) -> None:
                super().__init__(name=name)

        register(Core)(by_init_params())

        with pytest.raises(AttributeError, match="name"):
            encode(Model("a"))

    def test_a_converter_on_the_base_covers_every_subclass(self):
        class Core:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args, self.kwargs = args, kwargs

        class Model(Core):
            def __init__(self, name: str) -> None:
                super().__init__(name=name)

        register(Core)(lambda model: {"args": model.args, "kwargs": model.kwargs})

        assert encode(Model("a")) != encode(Model("b"))
        assert encode(Model("a")) == encode(Model("a"))

    def test_sibling_subclasses_with_equal_state_do_not_collide(self):
        class Core:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args, self.kwargs = args, kwargs

        class Model(Core):
            def __init__(self, name: str) -> None:
                super().__init__(name=name)

        class Other(Core):
            def __init__(self, name: str) -> None:
                super().__init__(name=name)

        register(Core)(lambda model: {"args": model.args, "kwargs": model.kwargs})

        assert encode(Model("a")) != encode(Other("a"))

    def test_positional_and_keyword_construction_give_different_keys(self):
        """A known property, not a bug. `super().__init__(name)` and `super().__init__(name=name)`
        land in different attributes, so one hierarchy must pick one and keep it.
        """

        class Core:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args, self.kwargs = args, kwargs

        class Positional(Core):
            def __init__(self, name: str) -> None:
                super().__init__(name)

        class Keyword(Core):
            def __init__(self, name: str) -> None:
                super().__init__(name=name)

        register(Core)(lambda model: {"args": model.args, "kwargs": model.kwargs})

        assert encode(Positional("a")) != encode(Keyword("a"))


class TestResolvedDefaults:
    """`by_init_params` reads the attribute, not the argument.

    A default resolved inside `__init__` therefore reaches the key, even though the caller passed
    `None`. A default resolved anywhere else does not, and a change to it is invisible.
    """

    def test_a_default_resolved_in_init_participates(self):
        class Metric:
            DEFAULT_LABELS = ["click"]

            def __init__(self, labels: list[str] | None = None) -> None:
                self.labels = Metric.DEFAULT_LABELS if labels is None else labels

        register(Metric)(by_init_params())
        before = encode(Metric())
        Metric.DEFAULT_LABELS = ["click", "purchase"]

        assert encode(Metric()) != before

    def test_an_explicit_value_equal_to_the_default_keys_the_same(self):
        class Metric:
            DEFAULT_LABELS = ["click"]

            def __init__(self, labels: list[str] | None = None) -> None:
                self.labels = Metric.DEFAULT_LABELS if labels is None else labels

        register(Metric)(by_init_params())

        assert encode(Metric()) == encode(Metric(["click"]))

    def test_a_default_resolved_outside_init_is_invisible(self):
        """The hazard. The attribute keeps `None`, so the key cannot see the constant change."""

        class Metric:
            DEFAULT_LABELS = ["click"]

            def __init__(self, labels: list[str] | None = None) -> None:
                self.labels = labels

            def resolved(self) -> list[str]:
                return Metric.DEFAULT_LABELS if self.labels is None else self.labels

        register(Metric)(by_init_params())
        before = encode(Metric())
        Metric.DEFAULT_LABELS = ["click", "purchase"]

        assert encode(Metric()) == before
