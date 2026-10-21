from __future__ import annotations

import functools
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, overload

from ediechain.core.node import Node, iter_nodes
from ediechain.core.runner import Runner
from ediechain.core.storage import Store
from ediechain.core.task import Task, TaskDef


class DuplicateTaskName(ValueError): ...


class AmbiguousTask(LookupError): ...


class NoRunner(RuntimeError): ...


def _answers(node: Node[Any], token: str) -> bool:
    return token == node.task.name or node.key.startswith(token)


def _ordered(terminals: tuple[Node[Any], ...]) -> tuple[Node[Any], ...]:
    order: dict[str, Node[Any]] = {}

    def visit(node: Node[Any]) -> None:
        if node.key in order:
            return
        for dependency in node.inputs:
            visit(dependency)
        order[node.key] = node

    for terminal in terminals:
        visit(terminal)

    by_name: dict[str, TaskDef] = {}
    for node in order.values():
        first = by_name.setdefault(node.task.name, node.task)
        if first != node.task:
            raise DuplicateTaskName(f"two tasks named {node.task.name!r} in one chain")

    return tuple(order.values())


@dataclass(frozen=True)
class Chain[R]:
    outputs: R
    nodes: tuple[Node[Any], ...]
    runner: Runner | None = None

    forced: frozenset[str] = frozenset()

    @classmethod
    def of(cls, *terminals: Node[Any], runner: Runner | None = None) -> Chain[tuple[Node[Any], ...]]:
        return Chain(outputs=terminals, nodes=_ordered(terminals), runner=runner)

    @property
    def terminals(self) -> tuple[Node[Any], ...]:
        return tuple(dict.fromkeys(iter_nodes(self.outputs)))

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[Node[Any]]:
        yield from self.nodes

    def __contains__(self, node: object) -> bool:
        return node in self.nodes

    def __str__(self) -> str:
        return "\n".join(f"{node.task.name} {node.key[:8]}" for node in self.nodes)

    @overload
    def __getitem__[T](self, target: Task[Any, T]) -> Node[T]: ...
    @overload
    def __getitem__(self, target: str) -> Node[Any]: ...
    def __getitem__(self, target: Task[Any, Any] | str) -> Node[Any]:
        if isinstance(target, str):
            return self._pasted(target)

        found = [node for node in self.nodes if node.task is target.definition]
        if not found:
            raise KeyError(target.definition.name)
        if len(found) > 1:
            raise AmbiguousTask(
                f"{target.definition.name} has {len(found)} nodes in this chain; pass the node instead"
            )
        return found[0]

    def _pasted(self, text: str) -> Node[Any]:
        # A token no node answers is noise — a mark, a branch — and is dropped. The rest narrow
        # together, so a name and a key pasted from one line select the node that satisfies both.
        tokens = [token for token in text.split() if any(_answers(node, token) for node in self.nodes)]
        if not tokens:
            raise KeyError(text)

        found = [node for node in self.nodes if all(_answers(node, token) for token in tokens)]
        if not found:
            raise KeyError(text)
        if len(found) > 1:
            lines = "\n  ".join(f"{node.task.name} {node.key[:8]}" for node in found)
            raise AmbiguousTask(f"{text!r} matches {len(found)} nodes; paste one of\n  {lines}")
        return found[0]

    @overload
    def value[T](self, target: Task[Any, T] | Node[T]) -> T: ...
    @overload
    def value(self, target: str) -> Any: ...
    def value(self, target: Task[Any, Any] | Node[Any] | str) -> Any:
        node = target if isinstance(target, Node) else self[target]
        return self._runner().value(node, self.forced)

    def run(self) -> None:
        self._runner().run(self, self.forced)

    def on(self, runner: Runner) -> Chain[R]:
        return replace(self, runner=runner)

    def force(self, *targets: Task[Any, Any] | Node[Any] | str) -> Chain[R]:
        keys = set(self.forced)
        for target in targets:
            node = target if isinstance(target, Node) else self[target]
            keys = keys.union(self._descendants(node))
        return replace(self, forced=frozenset(keys))

    def _descendants(self, node: Node[Any]) -> set[str]:
        consumers: dict[str, list[Node[Any]]] = defaultdict(list)
        for other in self.nodes:
            for dependency in other.inputs:
                consumers[dependency.key].append(other)

        reached, stack = set(), [node]
        while stack:
            current = stack.pop()
            if current.key in reached:
                continue
            reached.add(current.key)
            stack.extend(consumers[current.key])
        return reached

    def _runner(self) -> Runner:
        if self.runner is None:
            raise NoRunner("this chain has no runner: pass root= or runner= to @chain, or use Chain.on()")
        return self.runner

    @property
    def store(self) -> Store:
        return self._runner().store


@overload
def chain[**P, R](compose: Callable[P, R], /) -> Callable[P, Chain[R]]: ...
@overload
def chain[**P, R](
    *, root: Path | str | None = None, runner: Runner | None = None
) -> Callable[[Callable[P, R]], Callable[P, Chain[R]]]: ...
def chain(
    compose: Callable[..., Any] | None = None,
    /,
    *,
    root: Path | str | None = None,
    runner: Runner | None = None,
) -> Any:
    if root is not None and runner is not None:
        raise TypeError("pass root= or runner=, not both")
    bound = runner if runner is not None else Runner(root=root) if root is not None else None

    def decorate(compose: Any, /) -> Any:
        @functools.wraps(compose)
        def build(*args: Any, **kwargs: Any) -> Chain[Any]:
            produced = compose(*args, **kwargs)
            terminals = tuple(dict.fromkeys(iter_nodes(produced)))
            return Chain(outputs=produced, nodes=_ordered(terminals), runner=bound)

        return build

    return decorate(compose) if compose is not None else decorate
