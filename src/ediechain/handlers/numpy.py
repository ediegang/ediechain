from pathlib import Path

import numpy as np

from ediechain import TypeHandler, register_handler


class NumpyHandler(TypeHandler):
    extension: str = "npy"

    def save(self, value: np.ndarray, path: Path) -> None:
        np.save(path, value)

    def load(self, path: Path) -> np.ndarray:
        # `allow_pickle` stays off: an object array would otherwise make the artifact executable.
        return np.load(path)


register_handler(np.ndarray, NumpyHandler())
