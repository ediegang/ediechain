# TaskChain — an evaluation for the `ediechain` rewrite

**Subject:** [flowerchecker/taskchain](https://github.com/flowerchecker/taskchain) v1.5.0, MIT,
commit `f2e23b4` (2025-06-03). Local copy: `playground/taskchain/`.
**Purpose:** decide which concepts `ediechain` inherits, which it inherits but rebuilds, and which
it refuses.

> Throughout, file references are **relative to `playground/taskchain/`** — `task.py:150` means
> `playground/taskchain/src/taskchain/task.py:150`; `docs/` and `tests/` paths are that repo's, not
> this one's.

---

## 1. Verdict

TaskChain is built around one genuinely good idea, wrapped in a lot of machinery that has not aged
well.

**The idea:** a task's output is stored at a path derived from a hash of *everything that
determines it* — its own parameter values plus the hashes of its inputs, computed recursively
through the DAG. Two different experiment configs that happen to agree on the inputs of some
upstream task share that task's data on disk automatically, with no bookkeeping, no run IDs, and no
"is this stale?" question. The authors describe it in their own comment as "a kind of blockchain"
(`chain.py:530-533`), which is exactly right — and it is the same insight behind Nix, Bazel and
DVC. **This should be the foundation of `ediechain`.**

**The machinery:** the rest of the library is a dense layer of implicit behaviour — string-keyed
declaration blocks, fuzzy name resolution, attribute-access magic, execution triggered as a side
effect of reading a property, and two full passes of object construction to compute the hash above.
It is ergonomic in a Jupyter notebook and hostile everywhere else: no type checking, no IDE
navigation, no scheduler, no parallelism, non-local failure modes, and runtime-only errors.

**The gap that matters most:** the hash covers parameters and input hashes but **not the task's
code or any declared version**. Edit the body of `run()` and TaskChain serves you the old result
forever, silently. This is design choice from developers of the TaskChain, in our solution we want
to have option to be able to make code part of the hash too. So I assume that some decorator can
be usefull pattern here.

---

## 2. What we are grading

| | |
| --- | --- |
| Source | 3,395 LOC across 17 modules (`src/taskchain/`) |
| Tests | 3,213 LOC, 121 test functions (`tests/`) — near 1:1 with source |
| Largest modules | `chain.py` 669, `task.py` 512, `data.py` 427, `config.py` 339, `parameter.py` 308, `cache.py` 308, `utils/clazz.py` 278 |
| Runtime deps | h5py, pandas, PyYAML, networkx, seaborn, filelock, icecream, tabulate, tqdm, orjson (+ numpy, matplotlib transitively) — and `pytest` is a **runtime** dependency, not a dev one |
| Docs | mkdocs-material site: `index`, `tasks`, `configs`, `chains`, `testing`, `utils`, `cheatsheet`, `example` + mkdocstrings API reference |
| Example | `example/` — a complete IMDB movie-rating project: 3 task modules, 10 configs, 5 notebooks |
| CI | GitHub Actions, single job, Python 3.10, `pytest` only — no lint, no type check, no matrix |
| Python | `^3.9,<4.0` |

Notable absences: no type checker in CI, no `py.typed`, no concurrency tests anywhere in the suite,
no test for `utils/migration.py`, and the `notes` file at the repo root is empty (0 bytes).

---

## 3. Architecture

### 3.1 The layers

```
YAML/JSON config files
        │  uses:, tasks:, parameter values, {class:...} object definitions
        ▼
   Config ──────────► Context           config.py
        │             (overlay that rewrites parameter values)
        │
        ▼
     Chain ─── builds ──► networkx DiGraph        chain.py
        │   • recursively loads `uses` configs
        │   • imports task classes from `tasks:` import strings
        │   • wires input_tasks, assigns namespaces
        │   • SECOND PASS: re-creates every task with a TaskParameterConfig
        ▼
      Task ── declares ──► Meta (input_tasks, parameters, ...)   task.py
        │   • run() is the user's computation
        │   • .value triggers load-or-compute, recursively
        ▼
      Data ── persists ──► {base}/{group}/{task}/{hash}.{ext}    data.py
                           + {hash}.run_info.yaml + {hash}.log
```

`cache.py` is an entirely separate, orthogonal layer: a `@cached` decorator for memoizing *methods*
inside a task's implementation. It shares no code with the task-level persistence above.

### 3.2 The one code path that matters

Everything hinges on what happens when a user writes `chain.my_task.value`:

1. `Chain.__getattr__` (`chain.py:96`) → `Chain.get` → `_find_task_full_name` (`task.py:482`)
   resolves `"my_task"` by **fuzzy suffix matching** against every task name in the chain.
2. `Task.value` (`task.py:222`) → `Task.data` (`task.py:150`), a property that does four jobs:
   - constructs the `Data` handler and calls `init_persistence` with the task's path and hash;
   - if the file exists and the task is not forced → `data.load(...)` and return;
   - otherwise: initialise run-info, attach a per-run file log handler, call `run()`, detach,
     validate and store the result, save;
   - on exception: `data.on_run_error()`, drop `_data`, re-raise.
3. `_get_run_arguments` (`task.py:187`) inspects `run()`'s signature and, **for each argument name**,
   looks it up first in `input_tasks` then in `parameters`. Finding it in neither, or in both, is a
   `KeyError`. Reading an input task's value calls `.value` on it — which recurses into step 2.
4. `_process_run_result` (`task.py:323`) type-checks the return against the annotation and hands it
   to the `Data` object, which writes it to `{base_dir}/{group}/{task}/{hash}.{ext}`.

So **execution is depth-first recursion driven by attribute access**. There is no scheduler, no
topological pass, no place to hook progress, parallelism, or a dry run. This single structural
choice is responsible for a large share of the limitations below.

---

## 4. Core concepts

Each concept is stated implementation-independently — this is the part meant to survive into
`ediechain` — followed by a verdict.

### 4.1 Task + the `Meta` block

A task is a class with a `run()` method and an inner `class Meta` declaring its dependencies:

```python
class Movies(ModuleTask):
    class Meta:
        input_tasks = [AllMovies]
        parameters = [Parameter('from_year', default=None)]

    def run(self, all_movies, from_year) -> pd.DataFrame:
        ...
```

The **return annotation is load-bearing**: it selects the persistence handler (`task.py:46-87`).
A task with no annotation and no `Meta.data_type` fails at construction.

Names are derived, not declared: `MetaTask.slugname` (`task.py:30-39`) snake-cases the class name
and strips a trailing `_task`, so `FilteredDataTask` → `filtered_data`. `ModuleTask` and
`DoubleModuleTask` (`task.py:432-442`) additionally derive the task's *group* from the defining
module's name via `inspect.getmodule`, giving `group:name` or `package:module:name`.

The `Meta` vocabulary the code actually reads is exactly **eight keys** — verified by enumerating
every `meta.get(...)` / `in cls.meta` in `src/`:

| Key | Read at | Effect |
| --- | --- | --- |
| `input_tasks` | `chain.py:283` | dependencies: classes, name strings, or `~regex` patterns |
| `parameters` | `task.py:132` | list of `Parameter` objects resolved against the config |
| `name` | `task.py:31` | override the derived slug |
| `task_group` | `task.py:27,99` | override the group |
| `abstract` | `chain.py:198` | skip during wildcard task discovery |
| `data_type` | `task.py:49` | alternative to the return annotation |
| `data_class` | `task.py:70` | force a specific persistence handler |
| `ignore_return_type_mismatch` | `task.py:341` | skip the return type check (requires `InMemoryData`) |

Anything else in `Meta` is free-form and readable via `self.meta.x` — the example project uses this
for a poor man's class-level configuration (`column_name`, `train` flags) to share one `run()`
across subclasses.

**Verdict: keep the concept, redesign the mechanism.** A declarative dependency block is right. A
stringly-typed inner class is not: it is invisible to type checkers, silently tolerates typos (see
the `group` / `task_group` defect in §7), and gives IDEs nothing to navigate.

### 4.2 Config: parameters and DAG description in one file

A config is a YAML/JSON document that is simultaneously (a) a bag of parameter values, (b) the list
of task classes to instantiate (`tasks:` as import strings, `*` wildcard supported), and (c) the
list of other configs it depends on (`uses:`).

```yaml
tasks: movie_ratings.tasks.rating_model.*
uses: "{CONFIGS_DIR}/features/all.yaml"

test_ratio: .05
model:
  class: movie_ratings.models.sklearn.LinearRegressionRatingModel
  kwargs: {normalize: True, regularization: 1}
```

Three sub-mechanisms deserve naming:

- **`{PLACEHOLDER}` substitution** from `global_vars` (`config.py:226-227`), so configs are
  machine-independent.
- **Object definitions**: any dict with a `class` key is instantiated into a live object during
  config loading (`config.py:229-241` → `utils/clazz.py:218`). This is how a whole model
  implementation becomes a parameter value.
- **`excluded_tasks`**, the inverse of `tasks`, processed first (`chain.py:184`).

Parameter resolution scope is subtle and worth stating explicitly: a task resolves its parameters
against *the config that declared it*, not the top-level config. Declaring a parameter in the outer
config for a task that lives in a used sub-config raises `ValueError`
(`tests/test_chain.py:75-86`).

**Verdict: keep — with the dual role split.** "Experiments are data, not code branches" is the
right call, and the object-definition mechanism is what makes it work for ML. But one class being
the parsed document, the mutated overlay result, the live object graph, *and* the persistence
identity is too much (see §4.5 and §7).

### 4.3 `uses`, namespaces, contexts

- **`uses:`** composes configs into a dependency chain. The example's model config uses the
  features config which uses the movies config — three levels.
- **Namespaces** (`uses: "path.yaml as train"`) instantiate the *same* sub-pipeline more than once
  in one chain under different parameters, addressed as `train::dataset`. Nested namespaces
  compose with `::` (`chain.py:139-145`). This is the feature that separates TaskChain from simple
  DAG runners — you can build a chain containing two differently-parameterised copies of a
  pipeline and compare them in one place.
- **Contexts** (`config.py:254-339`) are overlay configs applied at instantiation time that rewrite
  parameter values globally (or per-namespace via `for_namespaces`). The example notebook uses one
  to swap a personal-ratings file across all configs in a `MultiChain` at once.

**Verdict: keep namespaces; keep contexts, scoped.** Namespaces earn their complexity. Contexts as
*global* overrides are a footgun the docs already warn about ("applied globaly... Be cearful with
parameters of the same name in different pipelines", `docs/configs.md:383-386`) — in a rewrite they
should be explicitly scoped by default.

### 4.4 Chain and MultiChain

`Chain` (`chain.py:35`) is the assembled DAG plus the user-facing facade: dict access, attribute
access, a `tasks_df` DataFrame, `draw()` via graphviz, `force()`, `dependent_tasks()` /
`required_tasks()` over a `networkx` graph, and `create_readable_filenames()` which symlinks
human-readable names next to the hashed files. The notebook ergonomics here are genuinely nice, and
`create_readable_filenames` in particular is a thoughtful answer to the "my data directory is full
of hashes" problem that content addressing creates.

`MultiChain` (`chain.py:582`) holds several chains that **share task instances** via a registry
keyed on `(slugname, hash)` — so identical upstream tasks are computed once and held in memory
once, across experiments.

**Verdict: keep both concepts.** Shared-instance multi-chain is the natural companion to content
addressing. Redesign the facade: see the `dict` subclassing and fuzzy-name problems in §6.

### 4.5 Parameter mode — content-addressed persistence ★

This is the crown jewel, and it is worth being precise about how it works.

After the chain is built normally, `_recreate_tasks_with_parameter_config` (`chain.py:208-233`)
walks the DAG bottom-up and **re-instantiates every task** with a `TaskParameterConfig`
(`chain.py:519-579`) — a synthetic config that carries only (a) the parameters this specific task
actually declares, and (b) the already-computed persistence hashes of its input tasks. The hash is
then:

```python
sha256(f'{parameter_repr}$$${input_tasks_repr}')[:32]      # chain.py:558-568
```

where `parameter_repr` is the sorted `name=repr(value)` of the task's own persistence-relevant
parameters, and `input_tasks_repr` is the sorted `name=hash` of its inputs. Recursion through the
DAG makes each hash transitively cover the entire upstream computation.

The payoff is demonstrated directly in `tests/test_chain.py:608-672`: two different configs that
differ only in parameters irrelevant to task `A` produce **one** file on disk in parameter mode and
**two** without it. Change a root parameter and every downstream artifact is invalidated
automatically because every downstream hash changes.

Two escape hatches on `Parameter` make this survivable in production:

- `ignore_persistence=True` — the parameter does not enter the hash at all (`verbose`, `n_threads`).
- `dont_persist_default_value=True` — the parameter enters the hash only when it differs from its
  default, so **adding a new parameter does not invalidate existing data**.

That second flag is hard-won operational wisdom. Without it, every new knob on a task costs you a
full recompute of terabytes. `AutoParameterObject` exposes the same two hatches per constructor
argument (`parameter.py:297-308`).

**Verdict: keep, as the core of the rewrite — and fix the three holes.** No code version in the key
(§7.1), a `repr()` string as the key material (§6), and non-atomic writes into a store that never
re-verifies (§7.2).

### 4.6 Data handlers

`Data` (`data.py:20`) is the persistence strategy, selected automatically by matching the `run()`
return type against each subclass's `DATA_TYPES` list (`task.py:68-87`, scanning `inheritors(Data)`
— so **registration is "having been imported"**, and two handlers claiming one type is an error).

| Handler | Types | On disk |
| --- | --- | --- |
| `JSONData` | `str, int, float, bool, dict, list` | `.json` |
| `NumpyData` | `np.ndarray` | `.npy` |
| `PandasData` | `DataFrame`, `Series` | `.pd` (pickle) |
| `FigureData` | `pylab.Figure` | pickle + `.png` + `.svg` |
| `GeneratedData` | `Generator` | `.jsonl` |
| `GeneratedDataLazy` | opt-in | `.jsonl`, returns a re-iterable callable |
| `DirData` | opt-in | a directory the task writes into itself |
| `ContinuesData` / `H5Data` | opt-in | a directory that **survives between runs** for resumable work |
| `InMemoryData` | opt-in | nothing — recomputed in every process |

Two design points stand out. `DirData` hands the task a `dir` to write into and only renames it
into place on success — the right pattern. `ContinuesData` deliberately keeps its temp directory
across runs so a long job can resume, with `finished()` as the explicit commit.

**Verdict: keep the concept and the type→handler dispatch; rebuild the implementations.** The
dispatch is good ergonomics. But the handler set hard-codes an opinion (pandas via pickle, figures
via pickle) into the core, and importing `data.py` drags in h5py, pandas, numpy, matplotlib and
pylab unconditionally. Handlers belong in a lazy registry with optional extras.

### 4.7 Parameters and parameter objects

`Parameter` (`parameter.py:71`) declares name, `dtype` (checked at set time), `default`,
`name_in_config` (alias), and the two persistence flags. `Path` gets special handling: config
carries a string, the task receives a `Path`.

`ParameterObject` (`parameter.py:228`) is the contract for *objects* used as parameter values: it
must implement `repr()` returning a string that uniquely identifies the object's construction,
because that string goes into the hash. `AutoParameterObject` (`parameter.py:263`) derives it
automatically by reflecting over `__init__`'s signature and reading `self.<arg>` or `self._<arg>`
off the instance — with an `AttributeError` if you forgot to store one, and this guard:

```python
assert 'object at 0x' not in args_repr, f'repr for arguments is fragile: {args_repr}'   # parameter.py:294
```

That assertion is the tell. The whole identity scheme rests on `repr()` strings being stable and
meaningful, and the library knows it, so it checks at runtime for the most common way that fails.

`InputTaskParameter` (`parameter.py:143`) is a small gem: an input task that is *optional*, falling
back to a default value when that task is not present in the chain, letting one task body work in
chains that do and don't contain an upstream stage.

**Verdict: keep `Parameter` and its flags; keep optional input tasks; drop repr-as-identity.**
See §8.2.

### 4.8 Run info and logs

Beside every artifact, TaskChain writes `{hash}.run_info.yaml` and `{hash}.log`
(`data.py:88-113`, `task.py:376-421`): the task's class and module, every parameter's repr, the
input-task hashes, the username, the taskchain version, start/end/duration, and anything the task
pushed via `self.save_to_run_info(...)`. `chain.my_task.run_info` reads it back.

**Verdict: keep, and extend.** Provenance sidecars are cheap and repeatedly save you when asking
"what actually produced this file?". The rewrite should add the code version and the resolved
environment.

### 4.9 The `@cached` decorator

`cache.py` is a self-contained method-memoization layer: `Cache` interface, `DummyCache`,
`InMemoryCache` (thread-local via `get_ident()`), `FileCache` + JSON/DataFrame/Numpy variants with
`filelock` locking, and a `@cached` decorator (`cache.py:233`) that builds keys from the callee's
bound arguments, supports `ignore_kwargs`, `force_cache`, `only_cache`, and — note — **`version`**
(`cache.py:247`), which the task layer lacks.

**Verdict: keep as an optional utility, out of the core.** It is useful and unrelated to the DAG;
it should not be in the same package namespace as the pipeline engine.

### 4.10 The testing harness

`utils/testing.py` provides `MockTask`, `TestChain`, and `create_test_task`:

```python
task = create_test_task(D, parameters={'pd': 3}, input_tasks={'a': 1, C: 2})
assert task.value == 6                                   # tests/test_testing.py:129-135
```

Crucially it builds a real `Chain` and runs the real `_process_dependencies` / `_build_graph`
wiring, only substituting `MockTask` for the mocked dependencies — so tests exercise production
semantics rather than a parallel fake. Misconfiguration still raises the same build-time errors.

**Verdict: keep, and treat it as a design constraint.** "You can unit-test one task with literal
values for its inputs, in one line, with no YAML" is a feature the rewrite should design *toward*
from the start, not bolt on.

---

## 5. Core classes at a glance

| Class | Location | Responsibility | Verdict |
| --- | --- | --- | --- |
| `MetaTask` | `task.py:20` | class-level name/group/data-type derivation | redesign |
| `Task` | `task.py:102` | user computation unit + load-or-compute lifecycle | split |
| `ModuleTask` / `DoubleModuleTask` | `task.py:432-442` | group from module path | keep (as opt-in) |
| `InputTasks` | `task.py:445` | dependency registry, by name or index | redesign |
| `Chain` | `chain.py:35` | config graph → task DAG; user facade | split |
| `TaskParameterConfig` | `chain.py:519` | **the content-addressed hash** | keep the idea |
| `MultiChain` | `chain.py:582` | many chains sharing task instances | keep |
| `Config` | `config.py:22` | parameters + DAG description + object graph | split |
| `Context` | `config.py:254` | overlay that rewrites parameter values | keep, scoped |
| `Parameter` | `parameter.py:71` | typed, defaulted, hash-flagged config binding | keep |
| `InputTaskParameter` | `parameter.py:143` | optional input task with fallback | keep |
| `ParameterObject` / `AutoParameterObject` | `parameter.py:228,263` | identity for object-valued params | redesign |
| `Data` + 12 subclasses | `data.py:20-428` | persistence strategy per type | keep concept |
| `Cache` / `@cached` | `cache.py:27,233` | method memoization (orthogonal) | keep, separate pkg |
| `TestChain` / `create_test_task` | `utils/testing.py` | task unit-testing harness | keep |

---

## 6. Keep / redesign / drop

### Keep

| Decision | Why it is right |
| --- | --- |
| Recursive content-addressed persistence | Correct invalidation with zero bookkeeping; cross-experiment sharing falls out for free |
| Configs as data, tasks as code | Experiments become reviewable files instead of code branches |
| Namespaces for repeated sub-pipelines | Enables "compare two parameterisations in one chain"; rare and valuable |
| Return-type → storage dispatch | `-> pd.DataFrame` and persistence is handled; low ceremony, extensible |
| `ignore_persistence` / `dont_persist_default_value` | Lets a pipeline evolve without invalidating existing data — real production wisdom |
| Optional input tasks with defaults | One task body works in chains with and without an upstream stage |
| Provenance sidecars next to artifacts | Cheap, and the thing you reach for when debugging a stale result |
| Resumable `ContinuesData` | Long jobs that survive interruption are a first-class need in ML |
| Shared task instances across chains | The natural companion to content addressing |
| One-line task unit-testing on real wiring | Sets the ergonomic bar for the rewrite |

### Redesign

**`Meta` as a stringly-typed inner class.** Typos are silent, values are untyped, `self.movies` is
unresolvable by any tooling. Use typed declarations (annotations/descriptors or a decorator with
typed arguments) that a type checker and an IDE can both see.

**`repr()` strings as hash material.** `parameter_repr` is literally `f'{name}={repr(value)}'`
joined with `###`. Cache identity therefore depends on the `repr()` stability of arbitrary objects
across library and Python upgrades — the `assert 'object at 0x' not in args_repr` guard
(`parameter.py:294`) exists precisely because this breaks. Use canonical serialisation (a canonical
JSON/msgpack encoding) over explicitly declared identity, and *refuse* to hash types you do not
know rather than hashing their `repr`.

**Name-based `run()` argument injection** (`task.py:187-208`). Clever, and it reads well, but:
argument names must match derived task slugs, so renaming a task class changes the *signature* of
every consumer; a name present in both parameters and inputs is a runtime `KeyError`
(`tests/test_task.py:345-355`); default arguments are not usable; and none of it is checkable
statically. Keep the readability, but resolve bindings from an explicit declaration.

**Fuzzy task-name resolution** (`task.py:482-512`). Matching by shortest unambiguous suffix, with a
tie-break that prefers a candidate that is a suffix of all others, is delightful in a notebook and
dangerous in a codebase: adding an unrelated task elsewhere in the project can make an existing,
untouched reference ambiguous and break it at runtime. **Non-local failure.** Resolve explicitly;
offer aliases; keep fuzzy matching as an interactive-only convenience.

**`Chain` and `Config` subclassing `dict` while storing nothing in it.** Both inherit `dict`, call
`super().__init__()`, then keep their real contents in `self.tasks` / `self._data` and override
`__getitem__`, `get`, and `__contains__` with incompatible semantics — `Chain.get(x, default)`
*raises* if you pass a default (`chain.py:102-105`). So `dict(config)` is empty, `**config` is
empty, and iteration yields nothing while `'x' in config` is true. That is a Liskov violation with
no upside. Use a plain class, or `Mapping` implemented honestly.

**`__getattr__` magic for task and parameter lookup.** `chain.my_task`, `config.my_param`,
`params.my_param` are pleasant in a notebook and cost you: no completion, no type checking,
collisions with real attributes, and confusing `AttributeError`s from inside `__getattr__`. Offer
it as an explicit opt-in facade over an honest API.

**The two-pass build** (`chain.py:114-127` → `208-233`). Every task is constructed, wired, thrown
away, re-constructed with a `TaskParameterConfig`, and re-wired — because the hash cannot be
computed until the graph exists. The fix is conceptual, not mechanical: separate an inert *task
spec* (class + resolved parameters + input refs, hashable) from a live *task instance*. Compute
hashes over specs; instantiate once.

**`Task.data` as a property** (`task.py:150-185`). Reading an attribute can run for hours, write to
disk, mutate a logger, and raise. It also mixes four responsibilities. Split into an explicit
executor that owns load/compute/save, and keep `.value` as sugar over it.

### Drop

| Thing | Location | Why |
| --- | --- | --- |
| `parameter_mode=False` | `chain.py:47` | A second persistence semantics keyed on config *name* — the docs themselves warn it breaks with contexts and config renames (`docs/chains.md:252-272`). One store, one semantics |
| `icecream.install()` | `__init__.py:12-14` | Importing the library injects `ic` into the host program's **builtins**. Advertised as a feature (`docs/utils.md:5`). A library must not mutate its host's builtins |
| Name-string `isinstance`/`issubclass` | `utils/clazz.py:111-143` | Shadows the builtins, compares fully-qualified names across the MRO. Self-labelled `# HACK: fix autoreload in jupyter notebooks`. Two unrelated classes sharing a dotted path become "the same" — a global correctness hazard traded for notebook convenience. Solve reloading at the registry layer |
| `pkg_resources.get_distribution` | `__init__.py:1-3` | Deprecated, slow at import, breaks in zipapps. `importlib.metadata` |
| Eager heavy imports in the core | `data.py:9-14` | h5py, pandas, numpy, pylab, matplotlib on any import. Core must be light; handlers lazy |
| `pytest` as a runtime dependency | `pyproject.toml` | Test tooling in production installs |
| Duplicate `parallel_map` | `utils/iter.py`, `utils/threading.py` | Two hand-rolled implementations of the same thing, both wrapping a `ThreadPoolExecutor` in an `asyncio` loop via the deprecated `get_event_loop()`, neither tested. `concurrent.futures` does this |
| `ReprStr.__reduce__` | `utils/data.py:107-109` | `eval()` during unpickling of a type that carries config-file-derived strings |

---

## 7. Defects found

Each was verified in source. Ordered by consequence.

### 7.1 No code version participates in the cache key — *critical*

`name_for_persistence` is `sha256(parameters ⊕ input_hashes)` (`chain.py:558-568`). `Meta` has no
`version` key (confirmed by enumerating every `meta.get(...)` in `src/`), and nothing hashes the
task's source.

**Failure:** you fix a bug in `run()`. Every downstream artifact keeps the old, wrong data forever.
Nothing warns you. The only remedy is remembering to `force()` the right tasks by hand — and
`force` state "is lost when chain is recreated" (`docs/chains.md:157`).

Ironically the *method*-level cache does have this: `@cached(version='v1')` (`cache.py:247`).

**Rewrite:** a declared `version` on every task, participating in the key. Manual bumping is the
safe default (source hashing invalidates on comment edits and cannot see imported helpers); a
source-hash mode can be opt-in.

### 7.2 Writes are not atomic — *high*

`JSONData.save` writes to the final path directly (`data.py:177`), as do `NumpyData` (`:192`),
`PandasData` (`:233`) and `FigureData` (`:248`).

**Failure:** kill the process mid-write (OOM, SIGKILL, node preemption). A truncated file is left
at the hash path. `exists()` returns `True`, so the content-addressed store treats it as a valid
result forever and never recomputes it. Poisoned cache, silent wrong data.

`DirData.save` (`data.py:354-358`) and `GeneratedDataLazy.save` (`:285-291`) *do* write to a temp
path and `shutil.move` into place — the pattern is known, just not applied uniformly.

**Rewrite:** every artifact written to a temp path and atomically renamed, without exception; a
completion marker for multi-file artifacts.

### 7.3 No locking on task data — *high*

`filelock` is a declared dependency but is used **only** in `cache.py:120,136`. Nothing guards
`Data.save`.

**Failure:** two processes (or two notebook kernels, the library's own primary use case) compute
the same task concurrently and interleave writes to one path. Combined with 7.2, the result is
corruption rather than merely wasted work. The test suite contains **zero** concurrency tests.

**Rewrite:** per-artifact lock plus atomic rename — compute-to-temp, lock, rename, release. Two
racing writers then cost duplicate work, never corruption.

### 7.4 Sentinel comparison uses `==` instead of `is` — *medium*

```python
return self.default == self.NO_DEFAULT          # parameter.py:41
if self.dont_persist_default_value and self.value == self.default:   # parameter.py:65
if self._value == self.NO_VALUE:                # parameter.py:114
```

**Failure:** a numpy array or DataFrame as a parameter value makes `value == NO_VALUE` return an
array, and the surrounding `if` raises `ValueError: The truth value of an array with more than one
element is ambiguous`. Worse, `parameter.py:65` decides *persistence identity* by `==` on a
user-supplied value — any type with a non-boolean or lenient `__eq__` can silently mis-key data.

**Rewrite:** `is` for sentinels; identity decisions on canonicalised bytes, never on `__eq__`.

### 7.5 `force(recompute=True)` has no topological order — *medium*

```python
for task in list(forced_tasks)[::-1]:       # chain.py:391
    _ = task.value
```

`forced_tasks` is a `set`. Reversing an arbitrary set ordering is not a topological sort. It
happens to work because `.value` recurses into dependencies on demand, so the *result* is correct —
but the ordering is meaningless, the intent is not, and any future non-recursive executor built on
this assumption breaks. `networkx.topological_sort` is already available.

### 7.6 `Data.value` cannot represent `None` — *medium*

```python
if not hasattr(self, '_value') or self._value is None:
    raise ValueError(f'Value of {self} is not set')      # data.py:77-79
```

**Failure:** a task that legitimately returns `None` (or a handler whose loaded value is `None`) is
indistinguishable from "not computed", and raises on read. Use a distinct sentinel.

### 7.7 Documentation contradicts the code — *medium (trust)*

`docs/tasks.md:37` documents a `Meta.group` key. The code reads **`task_group`**
(`task.py:27,99`). Setting `group` in `Meta` silently does nothing — the task lands in the default
group and its data lands in a different directory than intended. A user following the documentation
gets no error.

This is the clearest argument against stringly-typed declaration blocks: an unknown key cannot be
rejected, because unknown keys are a feature (the example project relies on them).

### 7.8 Hashing on unsubstituted placeholders — *nuance, not a bug*

Parameter reprs preserve the **raw** config text: `'{DATA_DIR}/x.csv'` hashes identically on every
machine (`ReprStr`, `utils/data.py:90-101`, exercised in `tests/test_parameter.py::test_global_vars`).

This is deliberate and mostly right — it is what lets a team share one data store across machines
with different mount points. But it also means two environments where that placeholder resolves to
*genuinely different data* collide on the same hash. The rewrite should keep the behaviour and make
it an explicit, documented property of the key, not an emergent one.

### 7.9 Smaller items

| Item | Location | Note |
| --- | --- | --- |
| Unclosed file handles | `config.py:87-89`, and throughout `data.py` | `json.load(path.open())` — relies on refcounting |
| `del task` as a "free" | `chain.py:249` | Deletes a local name; the object is returned to the registry path anyway. Misunderstands Python |
| Constructor skipping | `chain.py:537` | `TaskParameterConfig` calls `super(Config, self).__init__()` to skip `Config.__init__`. A sign the class shouldn't be a `Config` |
| Hostile `get` overrides | `chain.py:102`, `task.py:467`, `parameter.py:188` | `get(x, default)` raises `ValueError` when a default is passed |
| Dead code after `return` | `utils/threading.py:57-58` | `gc.collect()` is unreachable |
| No-op statement | `utils/json.py:7` | Bare `orjson.OPT_SORT_KEYS` expression |
| `@persistent` and `None` | `utils/clazz.py:43` | Caches on `is None`, so a method legitimately returning `None` recomputes every call |
| Approximate-equality migration guard | `utils/migration.py:49-53` | Compares pandas files by **file size** within a tolerance to decide identity; untested module |
| Stale example | `example/scripts/tasks_run.py:6` | References `movies/20210624.imdb.yaml`, which no longer exists — the script cannot run |
| Docs example missing `self` | `docs/tasks.md:21` | `def run(dependency, param)` — the canonical snippet in the docs is not valid |
| Empty `notes` | repo root | 0 bytes |

---

## 8. What the rewrite should do differently

Principles, with sketches. These are illustrative — the API is not being decided here.

### 8.1 Separate spec → plan → execution

Three things are conflated in `Task` today: the declaration, the resolved instance, and the engine
that runs it. Splitting them removes the double-instantiation pass, makes the DAG inspectable
before anything runs, and gives parallelism somewhere to live.

```python
TaskSpec    # inert: class ref, resolved params, input refs, version. Hashable. No I/O.
Plan        # the DAG of specs + computed keys. Answers "what would run?" without running.
Executor    # walks the plan in topological order: load | compute | store. Owns concurrency.
Handle      # what the user holds: handle.value triggers the executor, lazily.
```

The user-facing ergonomics (`chain.my_task.value`) can be preserved exactly — as a thin facade over
`Executor`, rather than as the mechanism itself.

### 8.2 Make the cache key explicit, canonical, and versioned

```python
key = blake2b(canonical_encode({
    "task":    "features:movies",
    "version": 3,                       # declared; bumped when run() changes  ← the missing piece
    "params":  {"from_year": 1945},     # canonical encoding, not repr()
    "inputs":  {"all_movies": "9f2c…"}, # recursive keys, sorted
}))
```

Rules: canonical encoding only; unknown types raise rather than falling back to `repr`; identity is
declared by the value's owner, never inferred from `__eq__`; the encoded key material is written
next to the artifact so a mismatch is *detectable* instead of silent.

### 8.3 Make every write atomic and every artifact locked

One code path, used by all handlers, no exceptions:

```
acquire lock(key) → compute into tmp/ → fsync → atomic rename → write sidecars → release
```

This closes 7.2 and 7.3 together and makes a shared network store safe for concurrent users, which
is how these pipelines are actually run.

### 8.4 Keep the core dependency-light; register handlers lazily

The engine should need no pandas, no h5py, no matplotlib, no networkx-at-import. Storage handlers
register through entry points or explicit registration and import their backend on first use. The
graphviz/seaborn drawing code belongs in an extra.

### 8.5 Typed declarations over string keys

Whatever the surface syntax, an unknown declaration key must be an error, a task reference must be
resolvable by an IDE, and a parameter's type must be visible to a checker. That alone eliminates
7.7 and a whole class of runtime `KeyError`s.

### 8.6 Explicit resolution, opt-in magic

Names resolve explicitly by default. Fuzzy matching, attribute access and wildcard task discovery
are available, clearly marked as interactive conveniences, and never the thing a committed pipeline
depends on.

### 8.7 Design for the test harness from day one

`create_test_task(MyTask, inputs={...}, params={...})` must exist in the first commit and must run
production wiring. TaskChain got this right; it is much harder to retrofit.

---

## 9. Open questions

These are deliberately left for the design phase:

1. **Task versioning: manual bump or source hash?** Manual is predictable but forgettable; source
   hashing catches edits but invalidates on whitespace and cannot see imported helpers. A hybrid
   (manual version, with an opt-in source-hash warning) is worth considering.
2. **Do namespaces earn their complexity?** They are the most powerful feature here and the source
   of the most intricate code (`::` composition, relative resolution, per-namespace contexts). An
   alternative is parameterised sub-chain instantiation with explicit prefixes.
3. **Is YAML still the right config surface**, or should configs be Python objects with a YAML
   loader on top — getting type checking for free and keeping files for the experiment matrix?
4. **How much notebook ergonomics to preserve?** The magic is genuinely productive for the people
   who use it daily. The question is whether it lives in the core or in an explicitly separate
   interactive facade.
5. **Does `ediechain` need a remote/shared artifact store** (S3/GCS) from the start? Content
   addressing makes this natural, and it changes the locking design.
