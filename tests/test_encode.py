"""Spec for the canonical encoder.

`encode` turns a value into a string. That string is the material the key is hashed from, so
two values that mean different things must never produce the same string, and a type the
encoder does not know must raise instead of falling back to `repr`.

Most tests assert relations between encodings. `TestFormat` pins the text itself, because people
read it to see what went into a key.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import Enum, IntEnum, StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple
from uuid import UUID

import pytest

from ediechain.core.converters import by_init_params
from ediechain.core.encode import EncodeError, encode
from ediechain.core.registry import register


def all_distinct(*values: Any) -> bool:
    return len({encode(value) for value in values}) == len(values)


class TestContract:
    def test_encode_returns_a_string(self):
        assert isinstance(encode({"a": [1, 2]}), str)

    def test_encoding_is_stable_across_calls(self):
        value = {"a": [1, (2, 3)], "b": {"c": None}}

        assert encode(value) == encode(value)

    def test_equal_values_built_separately_encode_the_same(self):
        assert encode({"a": [1, 2]}) == encode({"a": [1, 2]})

    def test_encode_error_is_a_type_error(self):
        assert issubclass(EncodeError, TypeError)


class TestPrimitives:
    def test_scalars_of_different_types_do_not_collide(self):
        assert all_distinct(None, True, False, 0, 1, "1", "True", "None", 1.0)

    def test_bool_does_not_collide_with_int(self):
        assert all_distinct(True, 1)
        assert all_distinct([False], [0])

    def test_int_does_not_collide_with_float(self):
        assert all_distinct(1, 1.0)

    def test_large_ints_keep_their_precision(self):
        assert all_distinct(2**70, 2**70 + 1)

    def test_floats_that_differ_only_in_the_last_bit_do_not_collide(self):
        assert all_distinct(0.1, 0.1 + 1e-17, 0.30000000000000004, 0.3)

    def test_negative_zero_does_not_collide_with_zero(self):
        assert all_distinct(0.0, -0.0)

    def test_float_specials_encode_and_stay_apart(self):
        assert all_distinct(float("inf"), float("-inf"), float("nan"))

    def test_two_nans_encode_the_same(self):
        assert encode(float("nan")) == encode(float("nan"))

    def test_unicode_survives(self):
        assert all_distinct("é", "e", "é", "é ")

    def test_bytes_do_not_collide_with_str(self):
        assert all_distinct(b"abc", "abc")


class TestContainers:
    def test_tuple_and_list_do_not_collide(self):
        assert all_distinct((1, 2), [1, 2])

    def test_set_and_list_do_not_collide(self):
        assert all_distinct({1, 2}, [1, 2], (1, 2), frozenset({1, 2}))

    def test_set_order_does_not_matter(self):
        assert encode({1, 2, 3}) == encode({3, 1, 2})

    def test_a_set_of_mixed_types_encodes(self):
        assert encode({1, "a", None}) == encode({None, "a", 1})

    def test_dict_insertion_order_does_not_matter(self):
        assert encode({"a": 1, "b": 2}) == encode({"b": 2, "a": 1})

    def test_int_and_str_dict_keys_do_not_collide(self):
        assert all_distinct({1: "a"}, {"1": "a"})

    def test_mixed_key_types_encode_deterministically(self):
        assert encode({1: "a", "b": 2}) == encode({"b": 2, 1: "a"})

    def test_a_dict_does_not_collide_with_its_items(self):
        assert all_distinct({"a": 1}, [("a", 1)], [["a", 1]])

    def test_empty_containers_do_not_collide(self):
        assert all_distinct([], (), {}, set(), frozenset(), "", b"", None, 0)

    def test_nesting_is_not_flattened(self):
        assert all_distinct([[1], [2]], [[1, 2]], [1, 2], [[[1, 2]]])

    def test_separators_cannot_be_injected_from_a_string(self):
        """A string that looks like the encoder's own punctuation must not forge a structure."""
        assert all_distinct(["a", "b"], ["a,b"], ['"a","b"'], ["a"], "ab")

    def test_a_key_cannot_be_forged_from_a_value(self):
        assert all_distinct({"a": "b:c"}, {"a:b": "c"}, {"a": {"b": "c"}})

    def test_a_dict_keyed_by_a_tag_does_not_collide_with_the_tagged_value(self):
        """Without the `dict` tag, a JSON object could be a map or a tagged value, and both of
        these would read `{"Recall":{"k":{"int":10}}}`.
        """

        class Recall:
            def __init__(self, k: int) -> None:
                self.k = k

        register(Recall, name="Recall")(by_init_params())

        assert all_distinct(Recall(k=10), {"Recall": {"k": 10}})

    def test_a_str_enum_key_does_not_collide_with_its_name(self):
        """A `StrEnum` member passes `isinstance(key, str)`, but its tag is not `str`."""

        class Color(StrEnum):
            RED = "RED"

        assert all_distinct({Color.RED: 1}, {"RED": 1})

    def test_a_str_subclass_key_does_not_collide_with_a_str_key(self):
        class Slug(str):
            pass

        assert all_distinct({Slug("a"): 1}, {"a": 1})

    def test_mixed_str_and_int_keys_do_not_collide_with_all_str_keys(self):
        assert all_distinct({"a": 1, 2: "b"}, {"a": 1, "2": "b"})


class TestFormat:
    """Every value is a one-key object `{tag: payload}`. A JSON object is a map only right after a
    `dict` tag, and only when every key is a `str`; any other dict keeps tagged pairs.
    """

    def test_a_scalar_is_its_tag_over_a_json_value(self):
        assert encode(None) == '{"none":null}'
        assert encode(True) == '{"bool":true}'
        assert encode(1) == '{"int":1}'
        assert encode(1.5) == '{"float":1.5}'
        assert encode(0.0) == '{"float":0.0}'
        assert encode("a") == '{"str":"a"}'

    def test_bytes_are_hex(self):
        assert encode(b"abc") == '{"bytes":"616263"}'

    def test_a_list_keeps_its_order(self):
        assert encode([1, True, "1"]) == '{"list":[{"int":1},{"bool":true},{"str":"1"}]}'

    def test_a_tuple_is_tagged_as_a_tuple(self):
        assert encode((1, 2)) == '{"tuple":[{"int":1},{"int":2}]}'

    def test_a_set_is_sorted_by_the_encoded_item(self):
        assert encode({3, 1, 2}) == '{"set":[{"int":1},{"int":2},{"int":3}]}'

    def test_a_dict_with_str_keys_is_an_object_sorted_by_key(self):
        assert encode({"b": 2, "a": 1}) == '{"dict":{"a":{"int":1},"b":{"int":2}}}'

    def test_an_empty_dict_is_an_empty_object(self):
        assert encode({}) == '{"dict":{}}'

    def test_a_dict_with_a_non_str_key_keeps_tagged_pairs(self):
        assert encode({1: "a"}) == '{"dict":[[{"int":1},{"str":"a"}]]}'

    def test_a_dict_with_mixed_keys_keeps_tagged_pairs_sorted_by_the_encoded_key(self):
        assert encode({"a": 1, 2: "b"}) == '{"dict":[[{"int":2},{"str":"b"}],[{"str":"a"},{"int":1}]]}'

    def test_a_string_is_not_escaped_to_ascii(self):
        assert encode("é") == '{"str":"é"}'

    def test_an_object_converted_by_init_params_reads_as_its_fields(self):
        class API:
            def __init__(self, model_name: str, temperature: float = 0.0) -> None:
                self.model_name, self.temperature = model_name, temperature

        register(API, name="API")(by_init_params())

        assert encode(API("gpt-5.4-mini")) == (
            '{"API":{"dict":{"model_name":{"str":"gpt-5.4-mini"},"temperature":{"float":0.0}}}}'
        )

    def test_a_tagged_value_inside_a_dict_keeps_both_tags(self):
        class Recall:
            def __init__(self, k: int) -> None:
                self.k = k

        register(Recall, name="Recall")(by_init_params())

        assert encode({"metric": Recall(k=10)}) == '{"dict":{"metric":{"Recall":{"dict":{"k":{"int":10}}}}}}'


class TestStdlibTypes:
    def test_path_does_not_collide_with_str(self):
        assert all_distinct(Path("/a/b"), "/a/b")

    def test_equal_paths_encode_the_same(self):
        assert encode(Path("/a/b")) == encode(Path("/a") / "b")
        assert encode(Path("/a/b/")) == encode(Path("/a/b"))

    def test_a_path_is_not_resolved(self):
        """Keys are shared through one mount point, so a path enters the key as written."""
        assert encode(Path("/a/../b")) != encode(Path("/b"))

    def test_a_relative_path_raises(self):
        """The same relative path opens a different file from a different working directory, so
        two runs would share one key over different data.
        """
        with pytest.raises(EncodeError, match="relative"):
            encode(Path("data/x.csv"))

    def test_a_relative_pure_path_raises(self):
        with pytest.raises(EncodeError, match="relative"):
            encode(PurePosixPath("data/x.csv"))

    def test_a_relative_path_nested_in_a_container_raises(self):
        with pytest.raises(EncodeError, match="relative"):
            encode({"inputs": [Path("/data/a.csv"), Path("data/b.csv")]})

    def test_a_relative_path_as_a_str_is_not_refused(self):
        """A `str` is not a path to the encoder. Joining it onto a root is the caller's choice."""
        assert encode("data/x.csv") == encode("data/x.csv")

    def test_enum_member_does_not_collide_with_its_value_or_name(self):
        class Color(Enum):
            RED = 1

        assert all_distinct(Color.RED, 1, "RED", "Color.RED")

    def test_enums_of_different_classes_do_not_collide(self):
        class Color(Enum):
            RED = 1

        class Shade(Enum):
            RED = 1

        assert all_distinct(Color.RED, Shade.RED)

    def test_int_enum_does_not_collide_with_its_int(self):
        """`IntEnum` is an `int` subclass. Dispatch must see the concrete type first."""

        class Level(IntEnum):
            LOW = 1

        assert all_distinct(Level.LOW, 1, True)

    def test_datetime_does_not_collide_with_date_or_str(self):
        assert all_distinct(
            datetime(2026, 1, 2, tzinfo=UTC),
            date(2026, 1, 2),
            "2026-01-02",
            "2026-01-02T00:00:00+00:00",
        )

    def test_timezone_participates(self):
        naive = datetime(2026, 1, 2, 3, 4)
        aware = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)

        assert all_distinct(naive, aware)

    def test_timedelta_does_not_collide_with_its_seconds(self):
        assert all_distinct(timedelta(seconds=60), 60, 60.0, timedelta(minutes=1, microseconds=1))

    def test_decimal_does_not_collide_with_float_or_str(self):
        assert all_distinct(Decimal("1.10"), Decimal("1.1"), 1.1, "1.1")

    def test_uuid_does_not_collide_with_its_str(self):
        value = UUID("12345678-1234-5678-1234-567812345678")

        assert all_distinct(value, str(value), value.hex)


class TestUnknownTypes:
    def test_a_bare_object_raises(self):
        with pytest.raises(EncodeError, match="cannot encode"):
            encode(object())

    def test_the_error_names_the_offending_type(self):
        class Widget:
            pass

        with pytest.raises(EncodeError, match="Widget"):
            encode(Widget())

    def test_an_unknown_type_nested_in_a_container_raises(self):
        with pytest.raises(EncodeError):
            encode({"models": [object()]})

    def test_a_readable_repr_is_not_an_encoding(self):
        class Pretty:
            def __repr__(self) -> str:
                return "Pretty(a=1)"

            def __str__(self) -> str:
                return "pretty"

        with pytest.raises(EncodeError):
            encode(Pretty())

    def test_probing_a_type_does_not_evaluate_its_properties(self):
        """Dispatch reads the class, not the instance. `hasattr` on an instance runs properties."""

        class Landmine:
            @property
            def slugname(self) -> str:
                raise AssertionError("the encoder evaluated a property while probing the type")

            @property
            def value(self) -> str:
                raise AssertionError("the encoder evaluated a property while probing the type")

        with pytest.raises(EncodeError):
            encode(Landmine())

    def test_a_function_raises(self):
        with pytest.raises(EncodeError):
            encode(all_distinct)


class TestSubclasses:
    def test_a_str_subclass_does_not_collide_with_its_value(self):
        class Slug(str):
            pass

        assert all_distinct(Slug("x"), "x")

    def test_a_dict_subclass_does_not_collide_with_its_value(self):
        class Config(dict):
            pass

        assert all_distinct(Config(a=1), {"a": 1})

    def test_a_named_tuple_does_not_collide_with_a_plain_tuple(self):
        class Point(NamedTuple):
            x: int
            y: int

        assert all_distinct(Point(1, 2), (1, 2), [1, 2], {"x": 1, "y": 2})


class TestDataclasses:
    def test_equal_dataclasses_encode_the_same(self):
        @dataclass(frozen=True)
        class Window:
            start: int
            end: int

        assert encode(Window(1, 2)) == encode(Window(1, 2))

    def test_field_values_participate(self):
        @dataclass(frozen=True)
        class Window:
            start: int
            end: int

        assert all_distinct(Window(1, 2), Window(2, 1), Window(1, 3))

    def test_a_dataclass_does_not_collide_with_a_dict_of_its_fields(self):
        @dataclass(frozen=True)
        class Window:
            start: int
            end: int

        assert all_distinct(Window(1, 2), {"start": 1, "end": 2}, (1, 2))

    def test_dataclasses_of_different_classes_do_not_collide(self):
        @dataclass(frozen=True)
        class Window:
            start: int

        @dataclass(frozen=True)
        class Span:
            start: int

        assert all_distinct(Window(1), Span(1))

    def test_field_values_are_encoded_recursively(self):
        @dataclass(frozen=True)
        class Job:
            out: Path

        assert all_distinct(Job(Path("/a")), Job(Path("/b")))

    def test_an_unencodable_field_raises(self):
        @dataclass(frozen=True)
        class Job:
            handle: Any

        with pytest.raises(EncodeError):
            encode(Job(object()))


class TestRegistry:
    def test_a_registered_type_encodes(self):
        class Money:
            def __init__(self, amount: int, currency: str) -> None:
                self.amount = amount
                self.currency = currency

        register(Money)(lambda m: {"amount": m.amount, "currency": m.currency})

        assert all_distinct(Money(1, "czk"), Money(2, "czk"), Money(1, "eur"))

    def test_a_registered_type_does_not_collide_with_its_conversion(self):
        class Money:
            def __init__(self, amount: int) -> None:
                self.amount = amount

        register(Money)(lambda m: {"amount": m.amount})

        assert all_distinct(Money(1), {"amount": 1})

    def test_registration_covers_subclasses_without_collapsing_them(self):
        class Money:
            def __init__(self, amount: int) -> None:
                self.amount = amount

        class Czk(Money):
            pass

        register(Money)(lambda m: {"amount": m.amount})

        assert encode(Czk(1)) == encode(Czk(1))
        assert all_distinct(Money(1), Czk(1))


class TestIsolationFromTheEnvironment:
    def test_encoding_does_not_depend_on_the_hash_seed(self):
        probe = "from ediechain.core.encode import encode; print(encode({'b': 1, 'a': [2, {'x': 'y'}], 'c': {1, 'k'}}))"
        outputs = {
            subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "PYTHONHASHSEED": seed},
            ).stdout
            for seed in ("0", "1", "12345")
        }

        assert len(outputs) == 1

    def test_a_backend_registers_itself_on_the_first_miss(self):
        """The lazy path, in a fresh process: nothing imports the backend by hand."""
        pytest.importorskip("numpy")
        probe = (
            "import numpy as np; from ediechain.core.encode import encode; "
            "print('ndarray' in encode(np.array([1, 2])))"
        )
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

        assert result.stdout.strip() == "True"

    def test_a_numpy_scalar_encodes_the_same_before_and_after_the_backend_loads(self):
        """`np.float64` subclasses `float`, so the MRO walk can answer before the backend is imported."""
        pytest.importorskip("numpy")
        probe = (
            "import numpy as np; from ediechain.core.encode import encode; "
            "before = encode(np.float64(1.5)); encode(np.zeros(1)); print(before == encode(np.float64(1.5)))"
        )
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

        assert result.stdout.strip() == "True"

    def test_importing_the_core_does_not_import_heavy_backends(self):
        probe = (
            "import sys, ediechain.core.encode as _; "
            "print([m for m in ('numpy', 'pandas', 'polars', 'h5py', 'matplotlib') if m in sys.modules])"
        )
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

        assert result.stdout.strip() == "[]"


class TestNumpy:
    def test_an_array_does_not_collide_with_a_list(self):
        np = pytest.importorskip("numpy")

        assert all_distinct(np.array([1, 2]), [1, 2], (1, 2))

    def test_equal_arrays_encode_the_same(self):
        np = pytest.importorskip("numpy")

        assert encode(np.array([[1, 2], [3, 4]])) == encode(np.array([[1, 2], [3, 4]]))

    def test_shape_and_dtype_participate(self):
        np = pytest.importorskip("numpy")

        assert all_distinct(
            np.array([1, 2, 3, 4]),
            np.array([[1, 2], [3, 4]]),
            np.array([1, 2, 3, 4], dtype=np.float64),
            np.array([1, 2, 3, 5]),
        )

    def test_a_numpy_scalar_does_not_collide_with_a_python_int(self):
        np = pytest.importorskip("numpy")

        assert all_distinct(np.int64(1), 1, np.float64(1))

    def test_an_array_of_nans_encodes(self):
        np = pytest.importorskip("numpy")

        assert encode(np.array([np.nan])) == encode(np.array([np.nan]))

    def test_an_object_array_is_refused(self):
        """Its bytes are addresses, so the key would change in every process."""
        np = pytest.importorskip("numpy")

        with pytest.raises(EncodeError):
            encode(np.array(["a", 1], dtype=object))


class TestPolars:
    def test_a_frame_does_not_collide_with_a_dict(self):
        pl = pytest.importorskip("polars")

        assert all_distinct(pl.DataFrame({"a": [1, 2]}), {"a": [1, 2]})

    def test_equal_frames_encode_the_same(self):
        pl = pytest.importorskip("polars")

        assert encode(pl.DataFrame({"a": [1, 2]})) == encode(pl.DataFrame({"a": [1, 2]}))

    def test_column_order_participates(self):
        pl = pytest.importorskip("polars")

        assert all_distinct(
            pl.DataFrame({"a": [1], "b": [2]}),
            pl.DataFrame({"b": [2], "a": [1]}),
            pl.DataFrame({"a": [2], "b": [1]}),
        )

    def test_a_series_does_not_collide_with_a_list(self):
        pl = pytest.importorskip("polars")

        assert all_distinct(pl.Series([1, 2]), [1, 2], (1, 2))

    def test_equal_series_encode_the_same(self):
        pl = pytest.importorskip("polars")

        assert encode(pl.Series("a", [1, 2])) == encode(pl.Series("a", [1, 2]))

    def test_the_series_dtype_participates(self):
        pl = pytest.importorskip("polars")

        assert all_distinct(pl.Series([1, 2]), pl.Series([1.0, 2.0]))

    def test_the_series_name_participates(self):
        pl = pytest.importorskip("polars")

        assert all_distinct(pl.Series("a", [1, 2]), pl.Series("b", [1, 2]))

    def test_a_series_does_not_collide_with_a_single_column_frame(self):
        pl = pytest.importorskip("polars")

        assert all_distinct(pl.Series("a", [1, 2]), pl.DataFrame({"a": [1, 2]}))
