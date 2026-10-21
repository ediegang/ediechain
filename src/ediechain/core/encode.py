from __future__ import annotations

import dataclasses
import json
import types
from collections.abc import Callable
from enum import Enum
from typing import Any

from ediechain.core.registry import converter_for, load_registrations, tag_for


class EncodeError(TypeError): ...


type Payload = None | bool | int | float | str | Tree | list[Tree] | list[list[Tree]] | dict[str, Tree]
type Tree = dict[str, Payload]


def encode(value: Any, encoder: Callable[[Any], Tree] | None = None) -> str:
    encoder = encoder or _tree
    return _text(encoder(value))


def _text(tree: Tree | list[Tree]) -> str:
    return json.dumps(tree, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dict(value: dict[Any, Any]) -> Payload:
    if all(type(key) is str for key in value):
        return {key: _tree(item) for key, item in value.items()}
    pairs = [[_tree(key), _tree(item)] for key, item in value.items()]
    return sorted(pairs, key=lambda pair: _text(pair[0]))


_BUILTINS: dict[type, tuple[str, Callable[[Any], Payload]]] = {
    types.NoneType: ("none", lambda _: None),
    bool: ("bool", bool),
    int: ("int", int),
    float: ("float", float),
    str: ("str", str),
    bytes: ("bytes", bytes.hex),
    list: ("list", lambda value: [_tree(v) for v in value]),
    tuple: ("tuple", lambda value: [_tree(v) for v in value]),
    set: ("set", lambda value: sorted((_tree(v) for v in value), key=_text)),
    frozenset: ("frozenset", lambda value: sorted((_tree(v) for v in value), key=_text)),
    dict: ("dict", _dict),
}
_LAZY: dict[str, str] = {"numpy": "ediechain.encoders.numpy", "polars": "ediechain.encoders.polars"}


def _foreign(value: Any) -> Tree:
    cls = type(value)
    tag = tag_for(cls)
    if (fn := converter_for(cls)) is not None:
        return {tag: _tree(fn(value))}

    # Before the MRO walk: a numpy scalar subclasses a builtin, so the walk would answer with the
    # builtin's encoding and the key would then depend on whether the backend was loaded earlier.
    if load_registrations(cls, _LAZY):
        return _tree(value)

    if isinstance(value, Enum):
        return {tag: _tree(value.name)}
    if dataclasses.is_dataclass(cls):
        return {tag: _tree({f.name: getattr(value, f.name) for f in dataclasses.fields(cls)})}

    for base in cls.__mro__[1:]:
        if (fn := converter_for(base)) is not None:
            return {tag: _tree(fn(value))}
        if (entry := _BUILTINS.get(base)) is not None:
            return {tag: entry[1](value)}

    raise EncodeError(f"cannot encode {tag}")


def _tree(value: Any) -> Tree:
    if (entry := _BUILTINS.get(type(value))) is not None:
        tag, build = entry
        return {tag: build(value)}
    return _foreign(value)
