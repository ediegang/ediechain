"""Spec for a store several people write to.

`mkdir` and `open` apply the process umask, so a directory one user creates can be unwritable for
the next — and under decision 19 a writer needs to create a directory inside `{root}/{task}`, not
just a file. This is the failure that arrives before any race does.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from ediechain.core.runner import Runner
from ediechain.core.storage import Store
from ediechain.core.task import task

GROUP_SHARED = 0o775      # what the team's taskchain store actually uses
GROUP_READABLE = 0o664
GROUP_SHARED_SETGID = 0o2775


@task
def numbers(seed: int) -> list[int]:
    raise NotImplementedError


@pytest.fixture
def restrictive_umask():
    """Nothing for the group, nothing for others. A common default, and the one that hurts."""
    previous = os.umask(0o027)
    yield
    os.umask(previous)


def commit(store: Store, node, value):
    store.commit(node, value, {"key": node.key})


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class TestDefaults:
    """A store is shared by default. The umask is what the defaults exist to override."""

    def test_the_default_directory_mode_is_group_writable(self, tmp_path: Path, restrictive_umask):
        store = Store(root=tmp_path)
        node = numbers(seed=1)
        commit(store, node, [1])

        assert mode(store.node_path(node).parent) == GROUP_SHARED
        assert mode(store.node_path(node)) == GROUP_SHARED

    def test_the_default_file_mode_is_group_writable(self, tmp_path: Path, restrictive_umask):
        store = Store(root=tmp_path)
        node = numbers(seed=1)
        commit(store, node, [1])

        assert mode(store.artifact_path(node)) == GROUP_READABLE
        assert mode(store.sidecar_path(node)) == GROUP_READABLE

    def test_a_runner_passes_the_same_defaults_down(self, tmp_path: Path):
        assert Runner(root=tmp_path).store.dir_mode == GROUP_SHARED
        assert Runner(root=tmp_path).store.file_mode == GROUP_READABLE

    def test_none_hands_the_decision_back_to_the_umask(self, tmp_path: Path, restrictive_umask):
        """The opt-out, for a store that must not be readable by the rest of the group."""
        store = Store(root=tmp_path, dir_mode=None, file_mode=None)
        node = numbers(seed=1)
        commit(store, node, [1])

        assert mode(store.node_path(node).parent) == 0o750
        assert mode(store.artifact_path(node)) == 0o640


class TestSharedStore:
    def test_the_task_directory_is_group_writable(self, tmp_path: Path, restrictive_umask):
        """The one that matters: a second writer creates a node *directory* here, not a file."""
        store = Store(root=tmp_path, dir_mode=GROUP_SHARED)
        node = numbers(seed=1)
        commit(store, node, [1])

        assert mode(store.node_path(node).parent) == GROUP_SHARED

    def test_the_node_directory_carries_the_mode_through_the_commit(self, tmp_path: Path, restrictive_umask):
        """The staging directory is renamed into place, so it has to be right before the rename."""
        store = Store(root=tmp_path, dir_mode=GROUP_SHARED)
        node = numbers(seed=1)
        commit(store, node, [1])

        assert mode(store.node_path(node)) == GROUP_SHARED

    def test_no_bit_is_added_that_was_not_asked_for(self, tmp_path: Path, restrictive_umask):
        """`0o775` means `0o775`. The store does not decide that a shared store wants setgid."""
        store = Store(root=tmp_path, dir_mode=GROUP_SHARED)
        node = numbers(seed=1)
        commit(store, node, [1])

        assert mode(store.node_path(node)) == GROUP_SHARED
        assert not mode(store.node_path(node)) & stat.S_ISGID

    def test_setgid_is_available_when_it_is_asked_for(self, tmp_path: Path, restrictive_umask):
        """Needed when writers do not share a primary group and the filesystem has no default ACL."""
        store = Store(root=tmp_path, dir_mode=GROUP_SHARED_SETGID)
        node = numbers(seed=1)
        commit(store, node, [1])

        assert mode(store.node_path(node)) & stat.S_ISGID

    def test_the_artifact_is_group_readable(self, tmp_path: Path, restrictive_umask):
        store = Store(root=tmp_path, dir_mode=GROUP_SHARED, file_mode=GROUP_READABLE)
        node = numbers(seed=1)
        commit(store, node, [1])

        assert mode(store.artifact_path(node)) == GROUP_READABLE

    def test_the_sidecar_is_group_readable(self, tmp_path: Path, restrictive_umask):
        store = Store(root=tmp_path, dir_mode=GROUP_SHARED, file_mode=GROUP_READABLE)
        node = numbers(seed=1)
        commit(store, node, [1])

        assert mode(store.sidecar_path(node)) == GROUP_READABLE

    def test_a_root_the_store_creates_gets_the_mode_too(self, tmp_path: Path, restrictive_umask):
        """`mkdir(parents=True)` makes the intermediates as well, and they are umask'd."""
        root = tmp_path / "shared" / "artifacts"
        store = Store(root=root, dir_mode=GROUP_SHARED)
        commit(store, numbers(seed=1), [1])

        assert mode(root) == GROUP_SHARED

    def test_an_existing_directory_is_left_alone(self, tmp_path: Path, restrictive_umask):
        """The store does not own `root`. An admin who set it up deliberately keeps their mode."""
        root = tmp_path / "preexisting"
        root.mkdir(mode=0o700)
        store = Store(root=root, dir_mode=GROUP_SHARED)
        commit(store, numbers(seed=1), [1])

        assert mode(root) == 0o700
