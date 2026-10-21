# Design notes

What was decided, and why each finished piece is built the way it is. `TODO.md` holds what is
left to do. `§` points at `docs/dev/taskchain_evaluation.md`.

## Decisions

| #  | Decision                                                   | Reason                                                     |
|----|------------------------------------------------------------|------------------------------------------------------------|
| 1  | Tasks are functions with `@task`, not classes with `Meta`. | Parameters are typed and visible to the IDE.               |
| 2  | A call expression builds the graph. It does not run it.    | Separates the spec from execution.                         |
| 3  | `chain()` takes terminal nodes and walks back.             | No global registry. No import side effects.                |
| 4  | Configs are Python modules. No YAML.                       | The experiment definition is type checked.                 |
| 5  | Experiment files compose tasks directly.                   | Equal arguments give equal keys, so repeated code is free. |
| 6  | `*args` and `**kwargs` are forbidden in task signatures.   | They break default handling and encoding.                  |
| 7  | Defaults are part of the key.                              | A changed default must invalidate the data.                |
| 8  | `transparent` and `ignore` opt out per parameter.          | A forgotten mark costs a recompute, not wrong data.        |
| 9  | No code hashing. `version=` is manual and optional.        | Decide later. Needs a key-vs-sidecar policy first.         |
| 10 | Node identity is structural, taken from `key`.             | Dedup in `chain()` is a set operation.                     |
| 11 | A value is encoded to a tagged tree, then to JSON.         | A string cannot forge a structure it is not.               |
| 12 | An unknown type raises. No `repr` fallback.                | A wrong key is worse than a failed run.                    |
| 13 | A non-builtin carries its class fullname in its tag.       | A subclass never collides with the base it walks to.       |
| 14 | Backends register themselves, imported on first miss.      | The core never imports numpy, polars or pandas.            |
| 15 | A type's tag is its class name, not its import path.       | A file move must not orphan every key that saw it.         |
| 16 | A task body need not be deterministic.                     | An LLM call is not reproducible. The key is the recipe.    |
| 17 | First writer wins; the loser adopts the stored value.      | An artifact's content never changes once written.          |
| 18 | `collective=True` marks a body that runs on every rank.    | Distribution is a property of the body, not the runner.    |
| 19 | A node commits as one directory, renamed into place.       | Artifact and sidecar land together or not at all.          |
| 20 | A handler opts into the return type with `typed`.          | A self-describing format must not pay for JSON's gap.      |
| 21 | JSON is msgspec, typed by the return annotation.           | The file stays plain; the signature puts the types back.   |

## The pieces

### Core primitives

- `TaskDef`: name, fn, signature, returns, version, `ignore`, `transparent`.
- `TaskDef` rejects `*args`/`**kwargs` and unknown marked parameters at construction.
- `TaskDef.bind()`: `Signature.bind` plus `apply_defaults`. Positional becomes keyword.
- `Node[T]`: frozen, structural identity, key computed in `__post_init__`.
- `map_nodes` and `iter_nodes`: list, tuple, Mapping. `str` is a leaf.
- `Node.inputs`: flat, deduped, order preserving.
- `compute_key`: recursive hash over task name, version, filtered arguments.

### Canonical encoder (`core/encode.py`)

- `encode(value) -> str`: a tagged tree, then one `json.dumps`. JSON does the escaping, so no
  string can forge a separator.
- A tree is a one-key object `{tag: payload}`. Scalars are JSON values: `{"int":1}`. The text is
  read by people, so `TestFormat` pins it.
- A JSON object is a map only right after a `dict` tag. Without that tag, `{"Recall": {"k": 10}}`
  and `Recall(k=10)` would read the same.
- A dict whose keys are all exactly `str` is a JSON object. Any other dict keeps tagged pairs.
  `type(key) is str`, not `isinstance`: a `StrEnum` key would collide with its name.
- Layer 1, terminal: `NoneType`, `bool`, `int`, `float`, `str`, `bytes`, `list`, `tuple`, `set`,
  `frozenset`, `dict`. Exact type lookup, so `bool` never falls into `int`.
- Layer 2, dispatch: exact converter → `Enum` → dataclass → MRO walk over both tables → raise.
  `Enum` runs before the MRO walk because `IntEnum` has `int` earlier in its MRO.
- Sets and tagged pairs sort on the encoded text, never the raw value. Mixed types do not compare.
- A non-builtin's tag is `cls.__qualname__`, with no module path (decision 15), so a file move, a
  package rename or a class defined in `__main__` does not orphan the data. `register(cls, name=)`
  overrides it for a class whose own name is not a stable enough identity.
- `_TAGS` maps tag to owning class. Two classes claiming one tag, derived or explicit, raise
  `AlreadyRegistered` at import time — the collision taskchain accepted silently. The check is per
  process; two repos against one store still need an explicit `name=`.
- `tag_for(cls)` is one function, used by the converter branch, the `Enum` branch and the dataclass
  branch alike.
- Registered from the stdlib: `datetime`, `date`, `timedelta`, `UUID`, `Decimal`.
- `PurePath` is registered in `converters.py`, because it raises `EncodeError`. An absolute path
  enters the key as written, never resolved. A relative one raises: the same text opens a
  different file from another working directory.
- `encoders/numpy.py` and `encoders/polars.py`, imported on the first dispatch miss, keyed on the
  root module name of the value's class. The entry is popped before the retry, so a miss cannot
  loop.
- `node.py` computes keys through `encode`, and registers a refusal for `Node`, so a node found in
  a container the traversal does not walk fails loudly instead of being hashed as a value.
- `by_init_params()` in `converters.py`: a converter that reflects over `type(obj).__mro__`, so one
  registration on a base covers a hierarchy and picks up what a subclass adds. This is the
  replacement for inheriting `AutoSlugname`, moved from a base class to one registration.

### `@task` (`core/task.py`)

- `@task` and `@task(name=, version=, ignore=, transparent=)`. Returns a `Task[**P, R]`, a frozen
  callable whose `__call__` binds and returns `Node[R]`. `.fn` is the original function, for
  calling a task body directly in a unit test.
- Typing option B'. `In[T] = T | Node[T]` annotates a parameter; `resolve(x)` narrows it back to
  `T` for the body and raises at run time if the executor did not substitute. Measured against
  `ty` and pyright: bodies check, wiring checks, a misspelled keyword is still caught. A and C
  both lost one of those three.
- `@overload` on the two decorator forms is load-bearing. Without it the return type is a union
  and every wiring line inherits it.
- Task identity is `fn.__name__`, overridable with `name=` — decision 15 one level up. A file move
  keeps the data.
- `returns` is resolved with `get_type_hints`, not `signature.return_annotation`, which holds the
  string `"str"` under `from __future__ import annotations`. A task with no return annotation is
  refused at decoration: storage would have nothing to dispatch on. `-> None` is distinguishable
  from a missing annotation and is allowed.
- `TaskDef.returns` is `Any`, not `type`. `list[str]` is not a `type`.

### `Chain` and `@chain` (`core/chain.py`)

- `@chain` decorates a function that composes tasks and returns nodes. It returns a `Chain[R]`,
  where `R` is whatever the function produced — a `Node`, or a `NamedTuple` of them. `Chain.of()`
  builds one from loose nodes.
- `Chain.outputs` is exactly what the function returned, so a `NamedTuple` names each terminal.
  That closes the multiple-outputs question at no cost; `iter_nodes` already walked tuples.
- DFS post-order over `node.inputs`, deduped by key. Frozen; `nodes`, `terminals`, `__len__`,
  `__iter__`, `__contains__`, and a `__str__` of one line per node.
- `chain[task]` looks a node up **by task identity**, not by name, and raises `AmbiguousTask` when
  one task has several nodes. `chain.value(task_or_node)` is generic, so the task's return
  annotation survives the lookup — a string key would throw it away.
- `@chain(root=...)` or `@chain(runner=...)` binds a default runner; `Chain.on(runner)` rebinds
  without rebuilding. A chain with no runner still prints and inspects, and raises `NoRunner` if
  asked for a value.
- `DuplicateTaskName`: two tasks sharing a name inside one chain are refused. The chain is the
  enforcement point, so task identity needs no global registry (decision 3).

### Storage (`core/storage.py`, `handlers/`)

- **Two handler protocols (decision 20).** `TypeHandler` is `save(value, path)` / `load(path)` —
  what parquet, pickle and anything self-describing implements. `TypedHandler` adds a `returns`
  argument and opts in with `typed: ClassVar[Literal[True]]`. `is_typed()` dispatches on the flag,
  not `isinstance`: a runtime-checkable Protocol matches on method names, and both have the same
  two. It returns `TypeIs`, so the checker narrows both branches.
- **A node commits as one directory (decision 19).** `Store.commit(node, value, info)` stages the
  artifact and `run_info.json` in `{root}/{task}/tmp.{token}/` and renames that directory onto
  `{root}/{task}/{key}/`. A directory rename is atomic, so no reader sees half a node, and the
  second writer gets `ENOTEMPTY` — first-writer-wins with no lock. `save`/`save_sidecar` are gone:
  an API that could write one without the other would reintroduce the state the decision removes.
  `ENOTEMPTY` is a plain `OSError`, so the catch tests `errno`, not `FileExistsError`.
- `Store.clean()` removes `tmp.*` directories a `SIGKILL` left behind and returns the count. There
  is no half-committed node to repair, so that is all the sweep has to do.
- **JSON is msgspec, typed by the return annotation (decision 21).** `save` guards with
  `msgspec.convert(value, returns)` and `load` decodes with `type=returns`. `dict[int, str]` and
  `list[tuple[int, int]]` round-trip; the file stays plain JSON (`{"1": "a"}`), so other tools
  still read it. A value that contradicts its annotation is refused before anything is written.
  `msgspec` is the package's only dependency.
- `Store.commit` returns the stored value, which a typed handler may have normalised. JSON has
  one number type, so an `int` under a `float` annotation is a legal widening that `convert`
  accepts; the writer continues with the widened value, or the cold run and the warm run disagree.
  Decision 17's "the loser adopts the stored value", applied to the winner.
- **A shared store takes `dir_mode` and `file_mode`.** `mkdir` and `open` apply the process
  umask, so without them a directory one user creates is unwritable for the next — and under
  decision 19 a second writer has to create a *directory* inside `{root}/{task}`, not just a
  file. `mkdir(mode=)` is itself masked, so the bits are set with `chmod` afterwards, and only on
  directories the store created: an existing root belongs to whoever set it up. `None` keeps the
  umask, so a single-user store is unchanged.
- A dataclass lands as an object, a `NamedTuple` as an array — so a persisted `NamedTuple` that
  gains a field changes the layout of every artifact already written.
- `handlers/pickle.py` is `PICKLE`, reachable only through `store=`. It is deliberately not in the
  table: registering it for `object` would make it the silent fallback, and an unpickled artifact
  resolves its classes by import path, which is decision 15 in reverse.
- `handler_for`: exact, then `get_origin` (so `-> list[int]` finds `list`), then the MRO, then the
  lazy backend, then `NoHandler`. Nothing is guessed — a dataclass return needs `store=JSON`.
  `msgspec.inspect.type_info` cannot stand in for the table: it answers "yes" for `pl.DataFrame`.
- `@task` resolves the handler at decoration time, so an unstorable return type fails at import.
- Existence of the node directory answers "is it computed", so a task returning `None` is a result
  and not an absence (§7.6).

### `Runner` (`core/runner.py`)

- `Runner(root=..., coordination=...)` builds its own `Store`. `Store` is infrastructure, not API.
- It **pulls**: `value(node)` returns a cached artifact and stops, so nothing above a cached node
  is touched. A forward sweep would recompute an unpersisted middle task on every warm run.
- `@task(store=IN_MEMORY)` marks a task whose result is never written — taskchain's
  `InMemoryData`. Those values are held in `Runner.values` for the runner's lifetime, because
  there is no artifact for a consumer to load. It is also the escape hatch for a return type no
  handler covers.
- `Coordination` is a three-method protocol: `should_write()`, `barrier()` and `agree(local)`.
  Only the write is restricted, never the body — a collective body must run on every rank, so a
  lock would deadlock it. No `filelock`: the atomic rename already makes racing writers cost
  duplicate work rather than corruption (§7.3), and a lock would not survive the `SIGKILL` that
  creates the half-state it is meant to prevent.
- `agree(local)` makes every rank take the same branch on "does this exist". A rank that answered
  alone would enter a collective body by itself and hang the job.
- After the barrier a rank that did not write adopts the stored value when the artifact is there.
  Decision 17's invariant — `runner.value(node)` returns what is in the store — has to hold for
  the ranks that computed as well as for the writer that lost the race.
- The sidecar is written inside `Store.commit`, so there is no order between it and the artifact
  to get wrong: task, version, key, arguments, input keys, user, start, end, duration, `forced`
  (§4.8). The module is deliberately absent — decision 15.

### Force (`Chain.force`)

- `chain.force(*targets)` returns a **new** chain whose `forced` holds the target's key and every
  descendant's. Built with `dataclasses.replace`, so a field added later cannot be dropped.
- The cascade is not optional: a forced node's key does not change, only its content, so every
  artifact below it would otherwise sit at a key that still looks valid.
- `node.inputs` points upstream, so `_descendants` inverts the edges once over `chain.nodes`.
- No topological sort. The runner pulls, so a forced node skips its `exists` check and everything
  below it recomputes on the way down. §7.5 is where taskchain reversed a `set` and called it a
  sort.
- `forced` lives on the chain (what), `Runner.recomputed` on the runner (whether it already
  happened), so a forced chain held in a notebook does the work once, not on every call.
- A forced save passes `overwrite=True`, so it replaces rather than raising `ArtifactExists`.
- `force_now` also bypasses `Runner.values`, or a forced `IN_MEMORY` node would return the stale
  cached value.
- The sidecar carries `"forced"` as a bool on every artifact, so the store can be asked which
  artifacts were forced.

### Environment

- `conftest.py` snapshots and restores the converter registry per test, and imports the lazy
  backends before taking the snapshot. Without it a test that registers a type poisons the next.
- `tests/test_end_to_end.py` runs one experiment twice, which is what proves the pieces compose.
- `examples/simple.py` runs the whole API: polars artifacts, an `IN_MEMORY` model, warm runs, a
  `NamedTuple` of terminals and the ambiguous-task error.
- `examples/distributed.py` is the accelerate example, run with `torchrun --nproc_per_node 2`. It
  cannot be called `accelerate.py`: torchrun puts the script's directory first on `sys.path`, so
  `import accelerate` would find the example. It probes the barrier once at startup and explains
  itself instead of crashing where distributed is unavailable — on macOS `accelerate` calls
  `torch.distributed.barrier(device_ids=[rank])` and torch builds the device from the machine
  accelerator, which is MPS even when the process group is gloo on CPU.
- `examples/demo/` is a repository that uses ediechain: domain code in `src/demo/{data,model,
  metrics}.py` with no ediechain import, `@task` wrappers in `src/demo/tasks.py` written once,
  experiments in `configs/` that compose and bind no runner, and a marimo notebook that picks one
  and a store. `configs` is importable because `[tool.marimo.runtime] pythonpath = ["."]` puts the
  project root back on `sys.path` — marimo inserts the *notebook's* directory, not the root.
- `ediechain/__init__.py` exports the public surface, so nothing reaches into `ediechain.core`.

## Rendering

Two layers, split by what actually needs a dependency, not by what feels like a feature.

**`core/render.py` — no dependencies, always available.** §6 forbids the core importing numpy,
polars or matplotlib. A string formatter is not a backend, and `print(chain)` must not depend on
an extra being installed.

Three views, because a tree cannot draw a DAG and only one of them has to.

| View                    | Direction          | For                                                |
|-------------------------|--------------------|----------------------------------------------------|
| `__str__`, flat list    | source → terminal  | reading order, logs, diffs. No convergence problem |
| `render.tree(chain)`    | terminal → sources | what this depends on, and what is stale            |
| `render.mermaid(chain)` | source → terminal  | the real DAG, with convergence drawn               |

- **The tree runs terminal → sources.** Rendered both ways against `examples/simple.py`: in data
  flow direction a node with two inputs can only sit under one of them, so the main path
  dead-ends in a reference and the heading becomes whichever source was visited first. Fan-in is
  every task with two inputs; fan-out is usually one cheap upstream node. Dependency direction
  puts the unavoidable reference where it costs least.
- A node already drawn appears as a reference line, never as a repeated subtree.
- `__str__` keeps the flat topological list and gains the state marks. It already reads in data
  flow order, which is the direction that felt right — it just cannot be a tree.
- `Chain._repr_markdown_` returns the mermaid.
- State comes from the chain's own runner when it has one (`@chain(root=...)` binds it), so
  `print(chain)` shows cached / to compute / in memory / forced for free. With no runner, shape
  only.
- Marks: `✓` on disk, `·` to compute, `○` never persisted, `!` forced or downstream of forced,
  `↑` drawn above.

**`visualize/` — behind `ediechain[visualize]`.** Only the parts that need a dependency.

- `print_chain(chain)` via `rich.tree.Tree`: colour by state, correct box drawing, terminal width
  and encoding handled. `__str__` stays plain — it ends up in logs, CI output and assertions.
- `dot(chain)` is also just a string, but it belongs with the renderer that consumes it.
- `render(chain, path)` writes an SVG or PNG.

`core/render.py` ships with the three views and the marks. `visualize/` does not exist yet —
see `TODO.md` → Deferred.

## Concurrency

taskchain is no guide here. `JSONData.save` is `json.dump(value, path.open('w'))` — straight to
the final path, no temp file — and `grep -c filelock data.py` is 0. Two overlapping writers
truncate and interleave into one path (§7.2, §7.3). It only ever worked because `Task.data`
checks `exists()` first, so a *non-overlapping* second run loads.

**Already safe.** Temp file, fsync, `replace`. No reader ever sees a partial artifact.

**First writer wins (decision 17).** After computing, check `exists` again; if it appeared and the
node is not forced, discard the local value and load the stored one. With non-deterministic bodies
this is not an optimisation, it is what keeps the store internally consistent: otherwise one
process writes `X_B` while another writes `Y_A` derived from `X_A`, and `Y`'s sidecar points at an
input whose content is gone. The invariant it buys: **`runner.value(node)` returns the content
that is in the store**, so two consumers of one artifact in one run cannot see different bytes.

**One directory per node, renamed into place (decision 19).** The artifact and its sidecar are two
files. Written in either order they leave a window, and the two windows are not equally bad:

- Sidecar first, artifact second: the writer that loses the race overwrites the winner's sidecar.
  The run_info then describes a run whose value was discarded. It parses, so nothing complains.
- Artifact first, sidecar second: a `SIGKILL` in between leaves an artifact with no run_info.
  A reader gets `FileNotFoundError` and knows it does not know.

A lock does not close either window. A lock excludes other writers; it does not survive the kill
that created the half-state.

The fix is to make the pair one commit. Stage both files in `{root}/{task}/tmp.{token}/`, then
`os.rename` that directory onto `{root}/{task}/{key}/`. A directory rename is atomic on POSIX, so
a reader sees the node complete or absent, never torn.

- **First writer wins comes for free.** The second rename fails with `ENOTEMPTY`, because the
  target directory is not empty. This replaces `os.link` on the artifact and keeps decision 17.
- **The layout changes.** `{root}/{task}/{key}.{ext}` becomes `{root}/{task}/{key}/data.{ext}`,
  with `run_info.json` beside it. The `artifact` entry in `docs/glossary.md` changes with it.
- **A forced overwrite is two renames.** The committed directory moves out to `tmp.{token}`, the
  new one moves in, the old one is removed. A reader inside that window sees the node *missing*,
  never torn, and a kill inside it leaves a `tmp.*` directory for the sweep below.
- **Cost: one directory per node.** Inode count doubles. The store already shards by task name;
  if a single task ever holds enough nodes to make one directory slow, shard on a key prefix the
  way taskchain's `FileCache` did.

**`Coordination.agree(local: bool) -> bool`. Done.** Every rank must take the same branch on
"does this exist", or one enters a collective body alone and the job hangs.
`SingleProcessCoordination` returns `local`; `AccelerateCoordination` sum-reduces and is true only
when *every* rank has the file. A broadcast from rank 0 would tell a machine without the artifact
to load one it does not have, which is the case `per_node=True` exists for.

**`@task(collective=True)`.** Default is *non*-collective: only the writing rank runs the body,
the others barrier and load. Eight ranks then make one LLM call, not eight — and a downstream
collective body gets one input, not eight different ones. A task opts in to running everywhere.

- `@task(store=IN_MEMORY)` is implicitly collective. There is no artifact for a non-writing rank
  to load, so every rank must compute it — which is affordable precisely because in-memory means
  cheap to recompute.
- A chain boundary remains the coarse-grained alternative: run the prep chain on the main rank,
  barrier, then run the training chain on all ranks, where everything upstream is already cached.
  The two compose — the flag decides who runs a body, the split decides when.
- If the writing rank raises, the others are left at the barrier. Standard for collective code,
  worth documenting rather than solving.

**`Accelerate` ships as `ediechain[accelerate]`.** `should_write` is `is_main_process`, or
`is_local_main_process` when the store is per-node scratch — `DirDataParallel` already
distinguished `GLOBAL_RANK` from `LOCAL_RANK`, and getting it backwards means either N duplicate
writes or a missing artifact.

**Stale temp files. Done.** `finally` does not run on `SIGKILL` — OOM, preemption, `scancel`.
Under decision 19 what accumulates is `{root}/{task}/tmp.{token}/` directories: a staged node that
never committed, or the old copy of a forced overwrite. `Store.clean()` removes them and returns
the count. It needs no repair pass for a half-committed node, because there is no such state.
Left: deciding who calls it, and when.

**Shared-store permissions. Done, and on by default.** `dir_mode=0o775` and `file_mode=0o664`
are the defaults on `Store` and `Runner` — the values the team's taskchain store already uses,
`drwxrwxr-x` on directories and `-rw-rw-r--` on files. A store is shared unless someone says
otherwise, so the umask is the thing being overridden, not the thing being followed. Measured
identical under `umask 022`, `027` and `077`, each of which otherwise produces something
different.

- No setgid. The store sets what it is given and adds nothing. The existing store does not use it
  because every writer has `users` as their primary group, so a new directory already lands in
  the right one. `0o2775` is there for a store whose writers do not share a group.
- The existing store also carries extended ACLs (the `+` in `ls -l`). On Linux the group bits of
  a `chmod` become the POSIX ACL **mask**, which caps named-user and named-group entries, so
  `0o775` leaves them intact while anything narrower would quietly clip them. Worth confirming on
  the real filesystem with `getfacl` before pointing a shared run at it — it was not reproducible
  on macOS.
- `dir_mode=None, file_mode=None` is the opt-out, for a store that must not be readable by the
  rest of the group. It restores plain umask behaviour.

## Traps

Each of these has been hit at least once.

- **`dataclasses.replace`, not manual construction.** `Chain` is built in four places; `on()`
  dropped `forced` twice.
- **`get_type_hints` resolves against the function's module only.** A task whose return type is
  defined inside a test function raises `NameError` from `<string>:1`.
- **A test that registers a type pollutes the next one.** `conftest.py` snapshots and restores the
  registry, and imports the lazy backends *before* taking the snapshot.
- **`tmp_path` is one directory per test.** Two runners built from it are the same store, which
  silently turns a "fresh store" assertion into a cache hit.
- **Equal arguments give the same node.** A test that wants one task to have two nodes must vary
  an argument.
- **`ENOTEMPTY` is a plain `OSError`.** `except FileExistsError` does not catch a lost rename
  race, and POSIX permits `EEXIST` too, so the catch tests `errno` against both.
- **Never `mkdir` the node directory.** A rename onto an *empty* directory succeeds, so creating
  the final path during staging turns first-writer-wins off with no error.
- **`typed = True` must be annotated** `ClassVar[Literal[True]]` on a handler that does not
  subclass `TypedHandler`, or it infers as `bool` and `ty` rejects it at the `store=` call site.
- **`TypeIs`, not `TypeGuard`,** for the handler dispatch. `TypeGuard` narrows only the true
  branch, so the two-argument call in the `else` stays a union and fails to check.
- **A `float` annotation accepts an `int`.** `msgspec.convert({"count": 20}, dict[str, float])`
  passes, so `Store.commit` returns the normalised value and the runner continues with it. A new
  typed handler must return from `save` for the same reason.
- **`mkdir(mode=...)` is masked by the umask.** `dir_mode` needs a `chmod` after the `mkdir`.
- **A handler that saves through a container format must unwrap on load.** `np.savez` then
  `np.load` returns an `NpzFile`, not the array. `NumpyHandler` writes `.npy` for that reason.
- **A lazy backend is registered twice, in two tables.** `encode._LAZY` maps a value's module to
  its encoder; `storage._LAZY` maps it to its handler. A backend added to one and not the other
  keys fine and then cannot be stored — `@task` resolves the handler while decorating, so the
  miss is a `NoHandler` at import.
- **Two ranks share one pipe.** `tests/test_accelerate.py` parses probe output by regex, not by
  line, because the ranks interleave mid-line.
- **marimo puts the *notebook's* directory on `sys.path`**, not the project root. `examples/demo`
  works because its `pyproject.toml` sets `[tool.marimo.runtime] pythonpath = ["."]`, so marimo
  has to be launched from the demo root and from its own venv.
- **A file named after a library shadows it.** `examples/accelerate.py` broke `import accelerate`
  under torchrun; it is `examples/distributed.py`.

## Sources

The rewrite was designed against these. Local copies live in `playground/`, which is gitignored.

- https://github.com/flowerchecker/taskchain/tree/master
- https://gitlab.seznam.net/research/pipetask
- https://gitlab.seznam.net/research/ediegang/-/tree/main/src/ediegang/taskchain
- https://gitlab.seznam.net/research/sreality-vision (`utils/taskchain/`)
