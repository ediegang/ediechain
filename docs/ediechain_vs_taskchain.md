---
title: "Coming from taskchain"
sub_title: "ediechain — what is different in practice"
theme:
    name: "catppuccin-latte"
---

<!-- alignment: center -->

![image:width:35%](../assets/logo.png)

**ediechain**

a task graph for research pipelines

<!-- end_slide -->

<!-- alignment: left -->

# Why we rewrote it

**taskchain stopped moving**

- Last commit `f2e23b4`, 2025-06-03. Newest release on PyPI is 1.4.0.
- `python = "^3.9,<4.0"`, CI runs one job on Python 3.10.
- `pandas >=1.5.1,<3.0.0`, `h5py`, `seaborn`, `matplotlib` are not optional.
- No polars handler — we wrote `PolarsDataFrame` ourselves, in `ediegang`.
- `pytest` is a runtime dependency.

**We use a quarter of it**

- Keep:
    - slugname
    - configs as experiments
    - return type → storage
    - sidecars
    - `ignore_persistence`
    - `dont_persist_default_value`
- Drop:
    - namespaces
    - contexts
    - `MultiChain`
    - `parameter_mode=False`
    - fuzzy name lookup
    - `@cached`,
    - `icecream` in builtins
    - the two-pass build.

<!-- end_slide -->

# Weight

`cloc`, 2026-10-12. taskchain v1.5.0 from `playground/taskchain/`.

|                         | taskchain               | ediechain                 |
|-------------------------|-------------------------|---------------------------|
| `src/`                  | 2,290 code lines        | 847 code lines            |
| the engine              | all of it               | `core/` — 652 lines       |
| four largest modules    | 449, 336, 319, 245      | 151, 135, 91, 87          |
| tests                   | 2,452 lines, 121 funcs  | 2,699 lines, 446 tests    |
| test lines per src line | 1.1                     | 3.2                       |
| runtime dependencies    | 11 declared, 30 pulled  | 1 declared, 1 pulled      |
| python                  | `^3.9`                  | `>=3.13`                  |


<!-- end_slide -->

# The map

| taskchain                           | ediechain                              |
|-------------------------------------|----------------------------------------|
| `class X(ModuleTask)` + `Meta`      | `@task` on a function                  |
| `input_tasks = [A, "b"]`            | arguments of the call expression       |
| `parameters = [Parameter(...)]`     | the function's own parameters          |
| `data_class = PolarsDataFrame`      | `-> pl.DataFrame`                      |
| `data_class = InMemoryData`         | `@task(store=IN_MEMORY)`               |
| `ignore_persistence=True`           | `@task(ignore=["client"])`             |
| `dont_persist_default_value=True`   | `@task(transparent=["clip_at"])`       |
| nothing                             | `@task(version=2)`                     |
| `AutoSlugname` base class           | `@converted_by(by_init_params())`      |
| `config.yaml`, `uses:`, `tasks:`    | a Python module that calls tasks       |
| `Config(...).chain()`               | `@chain` function → `Chain`            |
| `chain.my_task.value`               | `chain.value(my_task)`                 |
| `{hash}.parquet` + `.run_info.yaml` | `{key}/data.parquet` + `run_info.json` |


<!-- end_slide -->

# A task, before and after

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->

`ragnaros/tasks/data/dev_data.py`

```python
class DevData(ModuleTask):
    class Meta:
        input_tasks = [ChubbyDevData]
        parameters = [
            Parameter("select_columns",
                      dtype=list, default=None),
            Parameter("verify_data", dtype=bool,
                      default=True,
                      ignore_persistence=True),
        ]
        data_class = PolarsDataFrame

    def run(self, chubby_dev_data,
            select_columns) -> pl.DataFrame:
        data = chubby_dev_data.development
        if select_columns is None:
            return data
        return data.select(select_columns)
```

<!-- column: 1 -->

the same task

```python
@task(ignore=["verify_data"])
def dev_data(
    chubby: In[DevelopmentData],
    select_columns: list[str] | None = None,
    verify_data: bool = True,
) -> pl.DataFrame:
    data = resolve(chubby).development
    if select_columns is None:
        return data
    return data.select(select_columns)
```

- the signature **is** the parameter list
- `-> pl.DataFrame` picks the handler
- `ignore=` is checked against the
  signature, so a typo raises at import
- `dev_data.fn(chubby=frame)` in a unit
  test: no chain, no YAML

<!-- reset_layout -->

<!-- end_slide -->

# Where every `Meta` key went

| `Meta` key                    | ediechain                              |
|-------------------------------|----------------------------------------|
| `input_tasks`                 | arguments at the call site             |
| `parameters`                  | the signature                          |
| `name`                        | `@task(name="model")`                  |
| `task_group`                  | gone — the key is task name + args     |
| `abstract`                    | gone — no wildcard discovery           |
| `data_type`                   | the return annotation                  |
| `data_class`                  | the return annotation, or `store=`     |
| `ignore_return_type_mismatch` | gone — JSON is checked, parquet is not |


<!-- end_slide -->

# Wiring: names, or expressions

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->

```python
class InferenceDev(ModuleTask):
    class Meta:
        input_tasks = ["dev_data"]
        parameters = [
            Parameter("model", dtype=Model),
        ]
        data_class = PolarsDataFrame

    def run(self, dev_data, model):
        return model.prediction(dev_data)
```

- the string must match a derived slug
- rename the class, break every consumer
- a name in both `parameters` and
  `input_tasks` is a runtime `KeyError`

<!-- column: 1 -->

```python
@task
def inference(
    data: In[pl.DataFrame],
    model: Model,
) -> pl.DataFrame:
    return model.prediction(resolve(data))


# in the experiment file:
inference(dev_data(...), model=api)
```

- the argument **is** the node
- the parameter name is yours
- a wrong keyword is a type error, before
  anything runs

<!-- reset_layout -->

`In[T] = T | Node[T]`.

At the call site the parameter holds a `Node`, because the artifact
does not exist yet.

Inside the body `resolve(data)` gives the value — and raises if the
runner did not substitute, instead of returning something wrong.

<!-- end_slide -->

# The experiment file

```yaml
# configs/synthetic_data/prompt_evaluation/organic.few_shot.yaml
tasks:
  - ragnaros.tasks.inference.inference.*
model:
  class: ragnaros.model.api.API
  kwargs:
    model_name: "sellma2-small"
    pre_model_prep:
      class: ragnaros.model.pre_model_prep.base.RenderPromptPreModelPrep
      kwargs: {column: "injected_segment", template_path: "templates/few_shot.jinja2"}
```

```python
# configs/organic_few_shot.py
from demo.model.api import API
from demo.model.prep import RenderPrompt
# ...

MODEL = API(
    model_name="sellma2-small",
    pre_model_prep=RenderPrompt(column="injected_segment",
                                template_path=TEMPLATES / "few_shot.jinja2"),
)

@chain
def experiment(columns: list[str] | None = None) -> Node[pl.DataFrame]:
    data = dev_data(loader=ORGANIC_LOGS, select_columns=columns)
    return metrics(inference(data, model=MODEL))
```

Import paths become imports. `kwargs:` becomes a constructor call. `uses:` becomes an import or a
function argument.

A **misspelled key fails in your editor**, not after the data loads.

<!-- end_slide -->

# Objects as parameters

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->

```python
class Model(AutoSlugname, ABC):
    ...
```

- a base class every value inherits
- identity is a `repr()` string
- `assert 'object at 0x' not in repr`
  guards the common failure
- an unknown object hashes as its `repr`,
  so a new key in every process

<!-- column: 1 -->

```python
@converted_by(by_init_params())
class Model(ABC):
    ...
```

- one registration, no base class
- same reflection over `__init__` and MRO
- identity is a tagged tree
- an unknown type **raises**

<!-- reset_layout -->

```python
encode(API("gpt-5.4-mini"))
['API', ['dict', [[['str', 'model_name'], ['str', 'gpt-5.4-mini']],
                  [['str', 'temperature'], ['float', '0.0']]]]]

encode([1, True, "1"])
['list', [['int', '1'], ['bool', 'True'], ['str', '1']]]
```

The tag is the class name, never the import path: moving a file does not orphan the data.
JSON does the escaping, so a string cannot forge a structure, and `1`, `True` and `"1"` are
three different keys.

<!-- end_slide -->

# What changes a key

| mark                             | ediechain                        | in the key      |
|----------------------------------|----------------------------------|-----------------|
| `ignore_persistence=True`        | `@task(ignore=["client"])`       | never           |
| `dont_persist_default_value`     | `@task(transparent=["clip_at"])` | if not default  |
| —                                | `@task(version=2)`               | always          |

```python
@task(ignore=["verbose"], transparent=["clip_at"], version=2)
def features(data: In[pl.DataFrame], clip_at: float = 2.0,
             verbose: bool = False) -> pl.DataFrame: ...
```

- **A changed default is a new key.** taskchain kept the old one. The safe side is the
  default: a forgotten mark costs a recompute, never wrong data.
- `version` is manual. No code hashing: a comment edit must not invalidate a terabyte, and a
  hash of the body cannot see the helper it imports.
- `chain.force(target)` is the other half — same key, new content — and the cascade
  downstream is **not optional**, because those keys did not change either.

<!-- end_slide -->

# Build, look, run

```python
@chain
def experiment(seed: int = 0, clip_at: float = 2.0) -> Node[float]:
    data = features(raw_data(seed=seed), clip_at=clip_at)
    return accuracy(predictions(trained_model(seed=seed), data))

pipeline = experiment(seed=0).on(Runner(root=PIPELINE_DIR))
print(render(pipeline))
```

```text
· accuracy 8ca09720          ✓ in the store, will be loaded
└── · predictions b54506a4   · will be computed
    ├── ○ model 5b3a39fc     ○ IN_MEMORY, never written
    └── · features d0043e47  ! forced, recomputed anyway
        └── ✓ raw_data 956cb5e9
```

- Calling a task returns a `Node`. Nothing runs. `@chain` walks back from the terminals.
- The experiment binds no store. The notebook picks it with `.on(runner)`.
- `pipeline.value(features)` keeps the task's return type.
  `pipeline.value("· features d0043e47")` takes a line pasted from the tree.
- Views: `tree`, `flat`, `mermaid`.

<!-- end_slide -->

# Storage

| return annotation                       | lands in                 |
|-----------------------------------------|--------------------------|
| `pl.DataFrame`, `LazyFrame`, `Series`   | `data.parquet`           |
| `np.ndarray`                            | `data.npy`               |
| `str`, `int`, `float`, `bool`, `list`, `dict` | `data.json` (msgspec) |
| a dataclass                             | `@task(store=JSON)`      |
| anything else                           | `@task(store=PICKLE)`    |
| not storable on purpose                 | `@task(store=IN_MEMORY)` |

```text
taskchain                        ediechain
{base}/{group}/{task}/           {root}/{task}/{key}/
    {hash}.parquet                   data.parquet
    {hash}.run_info.yaml             run_info.json
    {hash}.log
written in place                 staged in tmp.*, then renamed
```

- The handler is resolved at decoration time: an unstorable return type fails at import, not
  three hours in.
- A directory rename is atomic. No half-written artifact that `exists()` calls valid, and the
  second writer is refused instead of interleaved.

<!-- end_slide -->

# Several processes, one store

```python
@task(collective=True)
def embeddings(data: In[pl.DataFrame]) -> np.ndarray: ...
```

```text
8 ranks ask for a node
   │
   ├─ in the store?  every rank agrees on the answer, then loads
   │
   ├─ plain task     rank 0 computes and writes, 7 wait at the barrier, then load
   │                 → 8 ranks make 1 API call, not 8
   │
   └─ collective     every rank enters the body: it is one distributed computation
```

- `uv add "ediechain[accelerate]"`, then `torchrun --nproc_per_node 2 ...`.
- Distribution is a property of the body, not of the runner.
- taskchain had no coordination and no locking on task data at all.

<!-- end_slide -->

# Migrating one pipeline

1. **Keep the domain code.** `Model`, `Metric`, the loaders, the preps — unchanged.
2. **Swap the base class for a registration.** `class Model(AutoSlugname)` →
   `@converted_by(by_init_params())` on `Model`.
3. **Task classes become functions.** `Meta.parameters` → parameters, `Meta.input_tasks` →
   `In[...]` parameters, `data_class` → the return annotation.
4. **Carry the flags over.** `ignore_persistence` → `ignore`,
   `dont_persist_default_value` → `transparent`.
5. **Each YAML becomes a module.** `class:`/`kwargs:` → constructor calls, `uses:` → an import
   or an argument, `{DATA_DIR}` → a `Path` from `ragnaros.config`.
6. **Run both.** The two stores are separate and the keys differ, so nothing collides — and
   the first ediechain run is cold.

<!-- end_slide -->

# Not there yet

| taskchain                   | status                                                 |
|-----------------------------|--------------------------------------------------------|
| namespaces (`as train`)     | not built — call the task twice with other arguments   |
| `Context` overlays          | not built — pass an argument to the experiment         |
| `MultiChain`                | not needed — sharing falls out of the key              |
| `chain.tasks_df`            | not built — leaderboards over N chains need a pattern  |
| readable symlinks           | deferred                                               |
| `@cached` on a method       | not ported — a separate concern                        |
| `DirData`, `ContinuesData`  | not built — no resumable multi-file artifact           |
| generator / `.jsonl` data   | not built                                              |
| `create_test_task`          | not built — `task.fn(...)` covers the simple case      |
| parallel execution          | deferred                                               |
| `{DATA_DIR}` in the key     | kept — an absolute `Path` enters as written            |
| `TaskBridge`                | open — cross-experiment references are undecided       |

<!-- end_slide -->

# Traps

- **A changed default invalidates.** Use `transparent` when it must not.
- **A `Path` enters the key as written.** Every machine must see the data at one absolute path,
  such as `/ceph/brno/...`. A relative `Path` raises. A symlink or `..` gives another key.
- **The key sees a path, not a file.** A file rewritten in place keeps its key, so the old
  artifact is loaded. Give new data a new file name.
- **A constant read inside a body is not in the key.** `labels = labels or DEFAULTS` hides an
  input. Resolve a default where it is stored, not where it is used.
- **A converter on a base class misses what a subclass adds** — unless it reflects over
  `type(obj)`, which `by_init_params()` does.
- **Passing a DataFrame as an argument hashes the whole frame.** Pass a node.
- `*args` and `**kwargs` in a task signature are refused: they break defaults and encoding.

<!-- end_slide -->

# Cheat sheet

```python
from ediechain import IN_MEMORY, JSON, In, Node, Runner, chain, render, resolve, task

@task                                   # name = function name
@task(name="model", version=2)          # rename; bump when the body changes meaning
@task(ignore=["client"])                # never in the key
@task(transparent=["clip_at"])          # in the key only when not default
@task(store=IN_MEMORY)                  # never written
@task(collective=True)                  # every rank runs the body

def step(data: In[pl.DataFrame], k: int = 3) -> pl.DataFrame:
    frame = resolve(data)               # In[T] → T inside the body

@chain                                  # composes; returns a Chain, runs nothing
def experiment(seed: int = 0) -> Node[float]: ...

pipeline = experiment(seed=0).on(Runner(root=ROOT))
print(render(pipeline))                 # tree | flat | mermaid
pipeline.value(step)                    # typed by the return annotation
pipeline.force("step").run()            # recompute, cascading down
```

`examples/demo/` — domain code, `@task` wrappers, configs, notebook. `docs/glossary.md` —
the words. `docs/dev/taskchain_evaluation.md` — the full review, with file and line.
