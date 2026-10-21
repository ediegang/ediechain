"""Spec for `AccelerateCoordination`, on two real CPU ranks.

`agree` decides whether every rank loads or every rank computes. With `per_node=True` each
machine has its own disk, so one rank can hold an artifact that another rank lacks.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys

import pytest

PROBE = """
from accelerate import Accelerator
from ediechain.coordinators.accelerate import AccelerateCoordination

accelerator = Accelerator(cpu=True)
coordination = AccelerateCoordination(accelerator, per_node=True)
rank = accelerator.process_index
print(f"everywhere {rank} {coordination.agree(True)}")
print(f"rank0-only {rank} {coordination.agree(rank == 0)}")
"""


@pytest.fixture(scope="module")
def agreed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, set[str]]:
    pytest.importorskip("accelerate")
    probe = tmp_path_factory.mktemp("accelerate") / "probe.py"
    probe.write_text(PROBE)
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    # Without the loopback address and interface, gloo resolves the hostname and takes minutes on macOS.
    loopback = "lo0" if sys.platform == "darwin" else "lo"
    result = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2",
            "--master_addr", "127.0.0.1", "--master_port", str(port), str(probe),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env={**os.environ, "GLOO_SOCKET_IFNAME": loopback},
    )
    # Both ranks share one pipe and their writes interleave inside a line, so the decisions are
    # matched wherever they land rather than read one per line.
    decisions: dict[str, set[str]] = {}
    reporting: dict[str, set[str]] = {}
    for case, rank, decision in re.findall(r"(everywhere|rank0-only) (\d+) (True|False)", result.stdout):
        decisions.setdefault(case, set()).add(decision)
        reporting.setdefault(case, set()).add(rank)

    # Without this a case that lost a rank's output would agree with itself and pass.
    assert reporting == {"everywhere": {"0", "1"}, "rank0-only": {"0", "1"}}, result.stdout
    return decisions


def test_every_rank_loads_when_every_rank_has_the_artifact(agreed: dict[str, set[str]]):
    assert agreed["everywhere"] == {"True"}


def test_a_rank_that_lacks_the_artifact_makes_every_rank_compute(agreed: dict[str, set[str]]):
    assert agreed["rank0-only"] == {"False"}
