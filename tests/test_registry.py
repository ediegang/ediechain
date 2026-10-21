"""Spec for `register`.

A registration says two things: how to convert a value, and what the resulting tree is tagged
with. The tag is the type's identity in every key that ever saw it, so it must survive a file
move, and two types must never share one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ediechain.core.encode import encode
from ediechain.core.registry import AlreadyRegistered, register


class TestDecorator:
    def test_register_returns_the_converter_unchanged(self):
        class Money:
            def __init__(self, amount: int) -> None:
                self.amount = amount

        def convert(money: Money) -> dict[str, Any]:
            return {"amount": money.amount}

        assert register(Money)(convert) is convert

    def test_a_second_registration_for_the_same_class_raises(self):
        class Money:
            def __init__(self, amount: int) -> None:
                self.amount = amount

        register(Money)(lambda m: {"amount": m.amount})

        with pytest.raises(AlreadyRegistered):
            register(Money)(lambda m: {"amount": m.amount})

    def test_the_duplicate_error_names_the_class(self):
        class Widget:
            pass

        register(Widget)(lambda _: {})

        with pytest.raises(AlreadyRegistered, match="Widget"):
            register(Widget)(lambda _: {})


class TestDefaultTag:
    def test_the_default_tag_does_not_carry_the_import_path(self):
        """A file move must not orphan every key the class ever produced."""

        class Money:
            def __init__(self, amount: int) -> None:
                self.amount = amount

        register(Money)(lambda m: {"amount": m.amount})

        assert Money.__module__ not in encode(Money(1))

    def test_a_dataclass_tag_does_not_carry_the_import_path_either(self):
        """The dataclass branch never sees `register`, so it needs the same derivation."""

        @dataclass(frozen=True)
        class Window:
            start: int

        assert Window.__module__ not in encode(Window(1))

    def test_two_classes_deriving_the_same_tag_cannot_both_register(self):
        class First:
            pass

        class Second:
            pass

        First.__qualname__ = "Money"
        Second.__qualname__ = "Money"
        register(First)(lambda _: {})

        with pytest.raises(AlreadyRegistered, match="Money"):
            register(Second)(lambda _: {})

    def test_a_derived_tag_and_an_explicit_name_share_one_namespace(self):
        class Taken:
            pass

        class Other:
            pass

        Taken.__qualname__ = "Coin"
        register(Taken)(lambda _: {})

        with pytest.raises(AlreadyRegistered, match="Coin"):
            register(Other, name="Coin")(lambda _: {})


class TestName:
    def test_a_named_type_is_tagged_with_its_name(self):
        class Money:
            def __init__(self, amount: int) -> None:
                self.amount = amount

        register(Money, name="mylib:Money")(lambda m: {"amount": m.amount})

        assert "mylib:Money" in encode(Money(1))

    def test_a_name_replaces_the_derived_tag(self):
        """`name=` is for a class whose own name is not a stable enough identity."""

        class Money:
            def __init__(self, amount: int) -> None:
                self.amount = amount

        register(Money, name="mylib:Money")(lambda m: {"amount": m.amount})

        assert Money.__qualname__ not in encode(Money(1))

    def test_a_name_still_separates_values(self):
        class Money:
            def __init__(self, amount: int) -> None:
                self.amount = amount

        register(Money, name="mylib:Money")(lambda m: {"amount": m.amount})

        assert encode(Money(1)) != encode(Money(2))

    def test_a_named_type_does_not_collide_with_its_conversion(self):
        class Money:
            def __init__(self, amount: int) -> None:
                self.amount = amount

        register(Money, name="mylib:Money")(lambda m: {"amount": m.amount})

        assert encode(Money(1)) != encode({"amount": 1})

    def test_two_classes_cannot_share_a_name(self):
        class Money:
            pass

        class Cash:
            pass

        register(Money, name="mylib:Money")(lambda _: {})

        with pytest.raises(AlreadyRegistered, match="mylib:Money"):
            register(Cash, name="mylib:Money")(lambda _: {})

    def test_the_name_is_optional(self):
        class Money:
            def __init__(self, amount: int) -> None:
                self.amount = amount

        register(Money)(lambda m: {"amount": m.amount})

        assert Money.__qualname__ in encode(Money(1))


class TestNameAndInheritance:
    def test_a_subclass_keeps_its_own_identity(self):
        """A name identifies one class. If a subclass inherited it, it would collide with its base."""

        class Base:
            def __init__(self, lr: float) -> None:
                self.lr = lr

        class Ridge(Base):
            pass

        register(Base, name="mylib:Base")(lambda m: {"lr": m.lr})

        assert encode(Base(1.0)) != encode(Ridge(1.0))
        assert "mylib:Base" not in encode(Ridge(1.0))

    def test_a_subclass_can_be_named_too(self):
        class Base:
            def __init__(self, lr: float) -> None:
                self.lr = lr

        class Ridge(Base):
            pass

        register(Base, name="mylib:Base")(lambda m: {"lr": m.lr})
        register(Ridge, name="mylib:Ridge")(lambda m: {"lr": m.lr})

        assert "mylib:Ridge" in encode(Ridge(1.0))
        assert encode(Base(1.0)) != encode(Ridge(1.0))
