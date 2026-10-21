from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ediechain.core.node import Node
from ediechain.core.storage import IN_MEMORY

if TYPE_CHECKING:
    from ediechain.core.chain import Chain
    from ediechain.core.runner import Runner

type View = Literal["flat", "tree", "mermaid"]

FORCED = "!"
MEMORY = "○"
STORED = "✓"
PENDING = "·"
REFERENCE = "↑"

SHORT_KEY = 8


def render(chain: Chain[Any], view: View = "tree") -> str:
    drawn = _VIEWS.get(view)
    if drawn is None:
        raise ValueError(f"unknown view {view!r}: pick one of {', '.join(_VIEWS)}")
    return drawn(chain)


def _mark(chain: Chain[Any], node: Node[Any], runner: Runner) -> str:
    if node.key in chain.forced:
        return FORCED
    # Before the store: `exists` answers False for an in-memory node whatever the runner has done.
    if node.task.store is IN_MEMORY:
        return MEMORY

    return STORED if runner.store.exists(node) else PENDING


def _marked(chain: Chain[Any], node: Node[Any]) -> str:
    runner = chain.runner
    if runner is None:
        return ""
    return f"{_mark(chain, node, runner)} "


def _label(chain: Chain[Any], node: Node[Any]) -> str:
    return f"{_marked(chain, node)}{node.task.name} {node.key[:SHORT_KEY]}"


def _flat(chain: Chain[Any]) -> str:
    return "\n".join(_label(chain, node) for node in chain.nodes)


def _tree(chain: Chain[Any]) -> str:
    lines: list[str] = []
    seen: set[str] = set()

    def draw(node: Node[Any], line_prefix: str, child_prefix: str, head: str = "") -> None:
        if node.key in seen:
            lines.append(f"{line_prefix}{REFERENCE} {head}{_label(chain, node)}")
            return
        # Marked before the inputs are walked: a diamond reaches this node twice from one branch.
        seen.add(node.key)
        lines.append(f"{line_prefix}{head}{_label(chain, node)}")

        inputs = node.inputs
        for index, dependency in enumerate(inputs):
            last = index == len(inputs) - 1
            draw(
                dependency,
                child_prefix + ("└── " if last else "├── "),
                child_prefix + ("    " if last else "│   "),
            )

    for head, terminal in _roots(chain):
        draw(terminal, "", "", head)
    return "\n".join(lines)


def _roots(chain: Chain[Any]) -> list[tuple[str, Node[Any]]]:
    names = getattr(type(chain.outputs), "_fields", None)
    if names is None or not all(isinstance(output, Node) for output in chain.outputs):
        return [("", terminal) for terminal in chain.terminals]
    return [(f"{name}: ", output) for name, output in zip(names, chain.outputs, strict=True)]


def _mermaid(chain: Chain[Any]) -> str:
    lines = ["graph TD"]
    lines += [f'    {_id(node)}["{_diagram_label(chain, node)}"]' for node in chain.nodes]
    lines += [f"    {_id(dependency)} --> {_id(node)}" for node in chain.nodes for dependency in node.inputs]
    return "\n".join(lines)


def _id(node: Node[Any]) -> str:
    # A key is hex, so a bare key can start with a digit, which mermaid refuses as an id.
    return f"n{node.key[:SHORT_KEY]}"


def _diagram_label(chain: Chain[Any], node: Node[Any]) -> str:
    # `@task(name=)` takes any string, and an unescaped quote closes the label.
    name = node.task.name.replace('"', "#quot;")
    return f"{_marked(chain, node)}{name}<br/>{node.key[:SHORT_KEY]}"


_VIEWS = {"flat": _flat, "tree": _tree, "mermaid": _mermaid}
