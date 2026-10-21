from __future__ import annotations

import importlib
from collections.abc import Callable
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID


class AlreadyRegistered(ValueError): ...


Converter = Callable[[Any], Any]

_CONVERTERS: dict[type, Converter] = {}
_TAGS: dict[str, type] = {}
_NAMES: dict[type, str] = {}


def register(cls: type, name: str | None = None) -> Callable[[Converter], Converter]:
    def decorator(fn: Converter) -> Converter:
        if cls in _CONVERTERS:
            raise AlreadyRegistered(f"type {cls.__module__}.{cls.__qualname__} is already registered")

        tag = name or cls.__qualname__
        if (tag_owner := _TAGS.get(tag)) is not None:
            raise AlreadyRegistered(
                f"tag {tag!r} is already taken by {tag_owner.__module__}.{tag_owner.__qualname__}"
            )

        _TAGS[tag] = cls
        _CONVERTERS[cls] = fn
        if name is not None:
            _NAMES[cls] = name

        return fn
    return decorator


def converted_by[T](
    converter: Callable[[T], Any] | str, *, name: str | None = None
) -> Callable[[type[T]], type[T]]:
    def decorate(cls: type[T]) -> type[T]:
        register(cls, name=name)(getattr(cls, converter) if isinstance(converter, str) else converter)
        return cls
    return decorate


def converter_for(cls: type) -> Converter | None:
    return _CONVERTERS.get(cls)


def load_registrations(cls: Any, table: dict[str, str]) -> bool:
    # True when a module was imported, so the caller knows to look the type up again. The entry is
    # removed, so a type its module does not register cannot ask twice and loop.
    root = getattr(cls, "__module__", "").split(".")[0]
    if (module := table.pop(root, None)) is None:
        return False
    importlib.import_module(module)
    return True


def tag_for(cls: type) -> str:
    return _NAMES.get(cls) or cls.__qualname__


register(datetime)(lambda value: value.isoformat())
register(date)(lambda value: value.isoformat())
register(timedelta)(lambda value: value.total_seconds())
register(UUID)(lambda value: value.bytes)
register(Decimal)(str)
