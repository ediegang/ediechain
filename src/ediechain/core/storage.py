import errno
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from secrets import token_hex
from typing import Any, ClassVar, Literal, Protocol, TypeIs, get_origin

import msgspec

from ediechain.core.node import Node
from ediechain.core.registry import AlreadyRegistered, load_registrations


class NoHandler(ValueError): ...


class ArtifactExists(FileExistsError): ...


class TypeHandler(Protocol):
    extension: str = ""

    def save(self, value: Any, path: Path) -> None:
        ...

    def load(self, path: Path) -> Any:
        ...


class TypedHandler(Protocol):
    """A format that cannot describe itself, so the store hands it the task's return annotation.

    Parquet carries its column types and pickle carries the class, so they implement `TypeHandler`.
    JSON has four scalars and string-only keys, so it cannot tell `20` from `20.0` or an int key
    from a string one without being told what the task promised.

    `save` returns the value a later `load` will give back. It is not always the value it was
    handed: an `int` under a `float` annotation is a legal widening, so the writer has to continue
    with the widened one or the cold run and the warm run disagree.
    """

    extension: str = ""
    typed: ClassVar[Literal[True]]

    def save(self, value: Any, path: Path, returns: Any) -> Any:
        ...

    def load(self, path: Path, returns: Any) -> Any:
        ...


type Handler = TypeHandler | TypedHandler


def is_typed(handler: Handler) -> TypeIs[TypedHandler]:
    # The flag, not `isinstance`: a runtime-checkable Protocol matches on method names, and both
    # protocols have `save` and `load`.
    return getattr(handler, "typed", False)


class JsonHandler(TypedHandler):
    extension = "json"
    typed = True

    def save(self, value: Any, path: Path, returns: Any) -> Any:
        try:
            stored = msgspec.convert(value, returns)
        except msgspec.ValidationError as e:
            raise TypeError(f"{value!r} does not match {returns}: {e}") from None
        path.write_bytes(msgspec.json.encode(stored))
        return stored

    def load(self, path: Path, returns: Any) -> Any:
        return msgspec.json.decode(path.read_bytes(), type=returns)


JSON = JsonHandler()


class _InMemoryHandler(TypeHandler):

    def save(self, value: Any, path: Path) -> None:
        raise NotImplementedError("in-memory handler does not support saving")

    def load(self, path: Path) -> Any:
        raise NotImplementedError("in-memory handler does not support loading")


IN_MEMORY: Handler = _InMemoryHandler()


_HANDLERS: dict[Any, Handler] = {}
_LAZY = {"numpy": "ediechain.handlers.numpy", "polars": "ediechain.handlers.polars"}


def register_handler(type_: type | Sequence[type], handler: Handler) -> None:
    types = type_ if isinstance(type_, Sequence) else (type_,)
    for type_ in types:
        if type_ in _HANDLERS:
            raise AlreadyRegistered(f"type {type_} already have handler")
        _HANDLERS[type_] = handler


def handler_for(returns: Any) -> Handler:
    for candidate in (returns, get_origin(returns)):
        if candidate is not None and (found := _HANDLERS.get(candidate)) is not None:
            return found

    if isinstance(returns, type):
        for base in returns.__mro__[1:]:
            if (found := _HANDLERS.get(base)) is not None:
                return found

    if load_registrations(returns, _LAZY):
        return handler_for(returns)

    raise NoHandler(f"no handler available for {returns!r}")


@dataclass(frozen=True)
class Store:
    root: Path
    dir_mode: int | None = 0o775
    file_mode: int | None = 0o664

    def _make_dir(self, path: Path) -> None:
        if self.dir_mode is None:
            path.mkdir(parents=True, exist_ok=True)
            return
        created = [p for p in (path, *path.parents) if not p.exists()]
        path.mkdir(parents=True, exist_ok=True)
        for directory in created:
            os.chmod(directory, self.dir_mode)

    def _handler(self, node: Node[Any]) -> Handler:
        return node.task.store or handler_for(node.task.returns)

    def _save(self, node: Node[Any], value: Any, path: Path) -> Any:
        handler = self._handler(node)
        if is_typed(handler):
            return handler.save(value, path, node.task.returns)
        handler.save(value, path)
        return value

    def node_path(self, node: Node[Any]) -> Path:
        return self.root / node.task.name / node.key

    def artifact_path(self, node: Node[Any]) -> Path:
        return self.node_path(node) / f"data.{self._handler(node).extension}"

    def sidecar_path(self, node: Node[Any]) -> Path:
        return self.node_path(node) / "run_info.json"

    def commit(self, node: Node[Any], value: Any, info: Mapping[str, Any], *, overwrite: bool = False) -> Any:
        path = self.node_path(node)
        self._make_dir(path.parent)

        staged = path.with_name(f"tmp.{token_hex(8)}")
        self._make_dir(staged)

        try:
            stored = self._save(node, value, staged / self.artifact_path(node).name)
            (staged / "run_info.json").write_text(json.dumps(info, default=str, indent=2), encoding="utf-8")

            for child in staged.iterdir():
                if self.file_mode is not None:
                    os.chmod(child, self.file_mode)
                _fsync(child)
            self._publish(staged, path, overwrite=overwrite)

        finally:
            shutil.rmtree(staged, ignore_errors=True)

        return stored

    def exists(self, node: Node[Any]) -> bool:
        if node.task.store is IN_MEMORY:
            return False
        return self.node_path(node).exists()

    def load(self, node: Node[Any]) -> Any:
        handler = self._handler(node)
        if is_typed(handler):
            return handler.load(self.artifact_path(node), node.task.returns)
        return handler.load(self.artifact_path(node))

    def load_sidecar(self, node: Node[Any]) -> Mapping[str, Any]:
        return json.loads(self.sidecar_path(node).read_text(encoding="utf-8"))

    def clean(self) -> int:
        removed = 0
        for staged in self.root.glob("*/tmp.*"):
            if staged.is_dir():
                shutil.rmtree(staged, ignore_errors=True)
                removed += 1
        return removed

    def _publish(self, staged: Path, final: Path, *, overwrite: bool) -> None:
        if not overwrite:
            try:
                os.rename(staged, final)
            except OSError as e:
                if e.errno in (errno.ENOTEMPTY, errno.EEXIST):
                    raise ArtifactExists(final.name) from None
                raise
            _fsync(final.parent)
            return

        previous = final.with_name(f"tmp.{token_hex(8)}")
        replaced = final.exists() and os.rename(final, previous) is None
        os.rename(staged, final)
        _fsync(final.parent)
        if replaced:
            shutil.rmtree(previous, ignore_errors=True)


def _fsync(path: Path) -> None:
    f = os.open(path, os.O_RDONLY)
    try:
        os.fsync(f)
    finally:
        os.close(f)


register_handler((str, int, float, bool, list, dict, type(None)), JSON)
