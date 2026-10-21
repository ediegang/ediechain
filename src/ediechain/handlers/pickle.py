import pickle  # noqa: S403
from pathlib import Path
from typing import Any

from ediechain import TypeHandler


class PickleHandler(TypeHandler):
    extension: str = "pkl"

    def save(self, value: Any, path: Path) -> None:
        path.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))

    def load(self, path: Path) -> Any:
        return pickle.loads(path.read_bytes())  # noqa: S301


PICKLE = PickleHandler()
