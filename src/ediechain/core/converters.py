from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from pathlib import PurePath
from typing import Any

from ediechain.core.encode import EncodeError, encode
from ediechain.core.registry import register


def by_init_params(
    *, exclude: Iterable[str] = frozenset(), transparent: Iterable[str] = frozenset(), skip_private: bool = True
) -> Callable[[Any], dict[str, Any]]:
    exclude = frozenset(exclude)
    transparent = frozenset(transparent)

    def convert(obj: Any) -> dict[str, Any]:
        params: dict[str, Any] = {}
        seen: set[str] = set()
        for _class in type(obj).__mro__:
            if _class is object:
                continue
            for name, param in inspect.signature(_class.__init__).parameters.items():
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD) or name in seen:
                    continue
                seen.add(name)
                if name in ("self", *exclude) or (skip_private and name.startswith("_")):
                    continue
                value = getattr(obj, name)
                if name in transparent and _holds_default(value, param.default):
                    continue
                params[name] = value
        return params
    return convert


def _holds_default(value: Any, default: Any) -> bool:
    if default is inspect.Parameter.empty:
        return False
    try:
        encoded_default = encode(default)
    except EncodeError:
        return False
    return encode(value) == encoded_default


@register(PurePath)
def _absolute_path(path: PurePath) -> str:
    if not path.is_absolute():
        raise EncodeError(f"cannot encode the relative path {str(path)!r}: make it absolute")
    return str(path)
