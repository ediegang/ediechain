from typing import Any

import numpy as np

from ediechain.core.encode import EncodeError
from ediechain.core.registry import register


def _checked(value: np.ndarray | np.generic) -> np.dtype:
    # An object array holds pointers, so its bytes are addresses that differ in every process and
    # the key would never hit the store.
    if value.dtype.hasobject:
        raise EncodeError(f"cannot encode an array of dtype {value.dtype}: its bytes are addresses")
    return value.dtype


def _array(value: np.ndarray) -> dict[str, Any]:
    return {"dtype": str(_checked(value)), "shape": value.shape, "bytes": value.tobytes()}


def _scalar(value: np.generic) -> dict[str, Any]:
    return {"dtype": str(_checked(value)), "bytes": value.tobytes()}


register(np.ndarray)(_array)
register(np.generic)(_scalar)
