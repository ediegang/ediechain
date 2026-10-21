from __future__ import annotations

import getpass
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ediechain.core.node import Node, key_arguments, map_nodes
from ediechain.core.storage import IN_MEMORY, ArtifactExists, Store

if TYPE_CHECKING:
    from ediechain.core.chain import Chain


class Coordination(Protocol):
    """How a runner behaves when several processes compute one chain.

    A collective task body runs on every rank, so a runner may not block one rank while another
    computes. Only the write is restricted, and `barrier` is where the ranks meet again.
    """

    def should_write(self) -> bool: ...

    def barrier(self) -> None: ...

    def agree(self, local: bool) -> bool: ...


class SingleProcessCoordination:
    def should_write(self) -> bool:
        return True

    def barrier(self) -> None:
        pass

    def agree(self, local: bool) -> bool:
        return local


@dataclass
class Runner:
    root: Path | str
    coordination: Coordination = field(default_factory=SingleProcessCoordination)
    values: dict[str, Any] = field(default_factory=dict)
    dir_mode: int | None = 0o775
    file_mode: int | None = 0o664
    store: Store = field(init=False)

    recomputed: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.store = Store(root=Path(self.root), dir_mode=self.dir_mode, file_mode=self.file_mode)

    def value[R](self, node: Node[R], forced: frozenset[str] = frozenset()) -> R:
        force_now = node.key in forced and node.key not in self.recomputed
        result = self._value(node, force_now, forced)
        if force_now:
            self.recomputed.add(node.key)
        return result

    def _value[R](self, node: Node[R], force_now: bool, forced: frozenset[str] = frozenset()) -> R:
        if not force_now and node.key in self.values:
            return self.values[node.key]

        persisted = node.task.store is not IN_MEMORY
        if not force_now and persisted and self.coordination.agree(self.store.exists(node)):
            return self.store.load(node)

        arguments = map_nodes(node.arguments, lambda dependency: self.value(dependency, forced))
        collective = node.task.collective or not persisted
        if not collective and not self.coordination.should_write():
            self.coordination.barrier()
            return self.store.load(node)

        started = datetime.now()
        result = node.task.fn(**arguments)
        ended = datetime.now()

        if persisted and self.coordination.should_write():
            try:
                result = self.store.commit(
                    node, result, self._info(node, started, ended, forced=force_now), overwrite=force_now
                )
            except ArtifactExists:
                result = self.store.load(node)

        self.coordination.barrier()
        if persisted and not self.coordination.should_write() and self.store.exists(node):
            result = self.store.load(node)
        if not persisted:
            self.values[node.key] = result

        return result

    def run(self, chain: Chain[Any], forced: frozenset[str]) -> None:
        for node in chain.terminals:
            self.value(node, forced=forced)

    def _info(self, node: Node[Any], started: datetime, ended: datetime, forced: bool) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        inputs: dict[str, str] = {}
        for name, value in key_arguments(node.task, node.arguments).items():
            if isinstance(value, Node):
                inputs[name] = value.key
            else:
                arguments[name] = map_nodes(value, lambda dependency: dependency.key)

        return {
            "task": {"name": node.task.name, "version": node.task.version},
            "key": node.key,
            "arguments": arguments,
            "inputs": inputs,
            "started": started.isoformat(),
            "ended": ended.isoformat(),
            "duration": (ended - started).total_seconds(),
            "user": getpass.getuser(),
            "forced": forced,
        }
