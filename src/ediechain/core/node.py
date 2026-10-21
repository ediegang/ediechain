from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from ediechain.core.encode import EncodeError, encode
from ediechain.core.registry import register

if TYPE_CHECKING:
    from ediechain.core.task import TaskDef


def map_nodes(value: Any, fn: Callable[[Node], Any]) -> Any:
    if isinstance(value, Node):
        return fn(value)
    # A container with nothing to substitute is returned as it came, so a type the rebuild below
    # cannot reconstruct still reaches the body whenever it holds no node.
    if next(iter_nodes(value), None) is None:
        return value
    if isinstance(value, tuple):
        mapped = [map_nodes(val, fn) for val in value]
        return type(value)(*mapped) if hasattr(value, "_fields") else type(value)(mapped)
    if isinstance(value, list):
        return type(value)([map_nodes(val, fn) for val in value])
    if isinstance(value, Mapping):
        return type(value)({k: map_nodes(v, fn) for k, v in value.items()})
    return value


def iter_nodes(value: Any) -> Iterator[Node]:
    if isinstance(value, Node):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_nodes(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from iter_nodes(item)


def key_arguments(task: TaskDef, arguments: Mapping[str, Any]) -> dict[str, Any]:
    params = task.signature.parameters
    keep = {}
    for name, value in arguments.items():
        if name in task.ignore:
            continue
        # A value that holds a node is never the default, and encoding it to compare would raise.
        if name in task.transparent and next(iter_nodes(value), None) is None:
            default = params[name].default
            if default is not inspect.Parameter.empty and encode(value) == encode(default):
                continue
        keep[name] = value
    return keep


def compute_key(task: TaskDef, arguments: Mapping[str, Any]) -> str:
    payload = {
        "task":    task.name,
        "version": task.version,
        "args":    map_nodes(key_arguments(task, arguments), lambda n: n.key),   # nodes → their keys, in place
    }
    return sha256(encode(payload).encode()).hexdigest()[:32]


@dataclass(frozen=True, eq=False)
class Node[T]:
    task: TaskDef
    arguments: Mapping[str, Any]
    key: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", compute_key(self.task, self.arguments))

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.key == other.key

    def __repr__(self) -> str:
        return f"<{self.task.name}#{self.key[:8]}>"

    @property
    def inputs(self) -> tuple[Node, ...]:
        return tuple(dict.fromkeys(iter_nodes(self.arguments)))


@register(Node)
def reject(*_, **__) -> Any:
    raise EncodeError("a node reached the encoder: it sits in a container the traversal does not walk")
