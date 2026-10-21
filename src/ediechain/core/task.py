from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, get_type_hints, overload

from ediechain.core.node import Node
from ediechain.core.storage import Handler, handler_for

type In[T] = T | Node[T]


def resolve[T](wired: In[T]) -> T:
    if isinstance(wired, Node):
        raise TypeError(f"{wired!r} was not substituted: resolve() ran outside the executor")
    return wired


@overload
def task[**P, R](fn: Callable[P, R], /) -> Task[P, R]: ...

@overload
def task[**P, R](
    *,
    name: str | None = None,
    version: int | None = None,
    ignore: Iterable[str] = (),
    transparent: Iterable[str] = (),
    store: Handler | None = None,
    collective: bool = False,
) -> Callable[[Callable[P, R]], Task[P, R]]: ...


def task(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    version: int | None = None,
    ignore: Iterable[str] = (),
    transparent: Iterable[str] = (),
    store: Handler | None = None,
    collective: bool = False,
) -> Any:
    def decorate(fn: Any, /) -> Task[..., Any]:
        try:
            hints = get_type_hints(fn)
        except NameError as e:
            raise NameError(
                f"task {name or fn.__name__!r}: failed to resolve type hints: {e}. "
                f"A task's annotations are resolved against its module, so a locally defined type does not work."
            ) from e

        if "return" not in hints:
            raise TypeError(f"task {name or fn.__name__!r} has no return annotation")

        return_hint = hints.get("return", None)
        if store is None:
            handler_for(return_hint)

        return Task(
            definition=TaskDef(
                fn=fn,
                signature=inspect.signature(fn),
                returns=return_hint,
                name=name or fn.__name__,
                version=version,
                ignore=frozenset(ignore),
                transparent=frozenset(transparent),
                store=store,
                collective=collective,
                ),
            fn=fn,
        )
    return decorate(fn) if fn is not None else decorate


@dataclass(frozen=True)
class TaskDef:
    fn: Callable[..., Any]
    signature: inspect.Signature
    returns: Any
    name: str
    version: int | None = None
    ignore: frozenset[str] = frozenset()
    transparent: frozenset[str] = frozenset()
    store: Handler | None = None
    collective: bool = False

    def __post_init__(self) -> None:
        for p in self.signature.parameters.values():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                raise TypeError(f"task {self.name!r}: *args/**kwargs are not allowed ({p.name!r})")

        unknown_params = (self.ignore | self.transparent) - set(self.signature.parameters)
        if unknown_params:
            raise ValueError(f"task {self.name!r}: unknown parameter(s) {sorted(unknown_params)}")

    def bind(self, *args: Any, **kwargs: Any) -> Node[Any]:
        # transform positional arguments into named
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        # MappingProxyType is read-only mapping
        return Node(task=self, arguments=MappingProxyType(dict(bound.arguments)))


@dataclass(frozen=True)
class Task[**P, R]:
    definition: TaskDef
    fn: Callable[P, R]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Node[R]:
        return self.definition.bind(*args, **kwargs)
